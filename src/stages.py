"""Execute retrieval and chunk selection with per-query checkpoints."""

import gc
import json

import numpy as np
import torch
from tqdm.auto import tqdm

from .checkpoint import RETRIEVAL_HASH, append_jsonl, load_jsonl

from .config import (
    CHECKPOINT_EVERY,
    CHUNK_ENCODE_BATCH,
    CHUNK_MIN_FREE_VRAM,
    CHUNK_TOPK,
)


def _restore_rows(checkpoint, path, legacy_name, is_compatible, *, chunks=False):
    rows = load_jsonl(path)
    from_legacy = False
    if not rows:
        legacy = checkpoint.result_path(legacy_name, chunks=chunks)
        if legacy is not None:
            try:
                rows = json.loads(legacy.read_text(encoding='utf-8'))
                from_legacy = True
                print('  resumed from the legacy file', legacy)
            except (OSError, ValueError) as exc:
                print('  could not read the legacy checkpoint:', exc)

    accepted = {str(row['id']): row for row in rows if is_compatible(row)}
    if from_legacy or len(accepted) != len(rows):
        # Migrate compatible legacy rows before later appends create a JSONL file.
        temporary = path.with_suffix('.tmp')
        temporary.write_text('', encoding='utf-8')
        append_jsonl(temporary, accepted.values())
        temporary.replace(path)
    return accepted


def run_retrieval(queries, retriever, checkpoint):
    RETRIEVAL_JSONL = checkpoint.materialize_result('retrieval.jsonl')
    RETRIEVAL_JSON = checkpoint.work_dir / 'retrieval_results.json'
    restored = _restore_rows(
        checkpoint, RETRIEVAL_JSONL, RETRIEVAL_JSON.name,
        lambda row: row.get('retrieval_hash') == RETRIEVAL_HASH,
    )
    active = {str(query['id']) for query in queries}
    retrieval_results = [row for qid, row in restored.items() if qid in active]

    print('retrieval rows restored:', len(retrieval_results))

    done_ids = {str(r['id']) for r in retrieval_results}
    print('{} queries left'.format(len(queries) - len(done_ids)))

    checkpoint.data['meta']['total_queries'] = len(queries)
    checkpoint.set_stage('retrieval', status='IN_PROGRESS', done=len(retrieval_results),
              total=len(queries), file='retrieval.jsonl')

    pending = []
    doc_counts = []
    failures = 0
    for q in tqdm(queries, desc='document retrieval'):
        if str(q['id']) in done_ids:
            continue
        try:
            top_docs, top_scores, _ = retriever.retrieve_documents(q)
        except Exception as exc:
            # ONE broken query (e.g. a CUDA OOM) must not kill the whole stage:
            # the row is kept empty and the run continues.
            print('  !! query {} failed: {}'.format(q['id'], exc))
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if 'out of memory' in str(exc).lower():
                # the GPU cannot take it any more -> continue on the CPU
                print('  !! CUDA OOM -> the dense retrieval continues on the CPU')
                retriever.embeddings.gpu_by_language.clear()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            top_docs, top_scores = [], []
            failures += 1

        row = {'id': q['id'], 'query': q.get('vi_orig', ''),
               'relevant_docs': top_docs,
               'scores': [round(float(s), 6) for s in top_scores],
               'n_docs': len(top_docs),
               'retrieval_hash': RETRIEVAL_HASH}
        retrieval_results.append(row)
        pending.append(row)
        doc_counts.append(len(top_docs))
        done_ids.add(str(q['id']))
        if len(pending) >= CHECKPOINT_EVERY:
            append_jsonl(RETRIEVAL_JSONL, pending)
            pending = []
            checkpoint.set_stage('retrieval', status='IN_PROGRESS', done=len(retrieval_results),
                      total=len(queries), file='retrieval.jsonl')
            # release the cached blocks of the GPU so the memory cannot creep up
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    append_jsonl(RETRIEVAL_JSONL, pending)
    RETRIEVAL_JSON.write_text(json.dumps(retrieval_results, ensure_ascii=False),
                              encoding='utf-8')
    checkpoint.set_stage('retrieval', status='DONE', done=len(retrieval_results),
              total=len(queries), file='retrieval.jsonl', json='retrieval_results.json')
    print('document retrieval done:', len(retrieval_results), 'queries')

    # --- report the distribution of the adaptive document counts ---------------- #
    if doc_counts:
        arr = np.asarray(doc_counts, dtype=float)
        print()
        print('adaptive document count over the {} new queries:'.format(len(arr)))
        print('  mean {:.1f} | median {:.0f} | min {:.0f} | max {:.0f}'.format(
            arr.mean(), float(np.median(arr)), arr.min(), arr.max()))
        print('  percentiles p10/p25/p50/p75/p90 = {}'.format(
            [int(np.percentile(arr, p)) for p in (10, 25, 50, 75, 90)]))
        hist = np.histogram(arr, bins=[0, 5, 10, 20, 40, 60, 80, 100, 10 ** 6])
        print('  histogram (count per bucket):')
        for lo, hi, c in zip(hist[1][:-1], hist[1][1:], hist[0]):
            print('    {:>4.0f}..{:<4.0f} {:>5}'.format(lo, hi, c))
    if failures:
        print('  !! {} query/queries failed and produced an empty result'.format(failures))

    return retrieval_results


def run_chunk_selection(retrieval_results, queries_by_id, selector, checkpoint):
    CHUNKS_JSONL = checkpoint.materialize_result('chunks.jsonl', chunks=True)
    CHUNK_JSON = checkpoint.work_dir / 'chunk_checkpoint.json'
    active = {str(row['id']): row['relevant_docs'] for row in retrieval_results}

    def matches_retrieval(row):
        qid = str(row['id'])
        return qid not in active or row['relevant_docs'] == active[qid]

    restored = _restore_rows(
        checkpoint, CHUNKS_JSONL, CHUNK_JSON.name, matches_retrieval, chunks=True,
    )
    submission = [row for qid, row in restored.items() if qid in active]
    print('chunk rows restored:', len(submission))

    done_qids = {str(s['id']) for s in submission}
    print('{} queries left'.format(len(retrieval_results) - len(done_qids)))

    checkpoint.set_stage('chunk', status='IN_PROGRESS', done=len(submission),
              total=len(retrieval_results), file='chunks.jsonl')

    # --- the chunk stage must survive a CUDA OOM ------------------------------- #
    # A CUDA OOM used to abort the whole notebook (after ~600 of the 1200
    # queries).  Now:
    #   * the batch is halved and the query is retried
    #   * the reduced batch STICKS for the following queries (it is never
    #     increased again) -> a query that OOMs is not repeated on every query
    #   * if even a batch of 1 does not fit, the dense scoring is given up for
    #     the rest of the stage: the BM25 score alone still selects the chunks
    #     (a bit weaker, but the query is not lost)
    # A single query can therefore NEVER kill the stage.
    CHUNK_BATCH = [CHUNK_ENCODE_BATCH]      # the working batch size
    CHUNK_DENSE = [True]                    # False -> BM25 only
    CHUNK_OOM_COUNT = [0]


    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass

    _free, _total = torch.cuda.mem_get_info()
    _free_gb = _free / 1024 ** 3
    print('GPU free/total: {:.2f} / {:.2f} GB'.format(_free_gb, _total / 1024 ** 3))

    if _free_gb < CHUNK_MIN_FREE_VRAM:
        print('  !! only {:.2f} GB free -> the chunk batch is reduced to {}'.format(
            _free_gb, max(1, CHUNK_ENCODE_BATCH // 2)))
        CHUNK_BATCH[0] = max(1, CHUNK_ENCODE_BATCH // 2)

    pending = []
    chunk_counts = []
    dup_total = 0
    for ret in tqdm(retrieval_results, desc='chunk selection'):
        qid = str(ret['id'])
        if qid in done_qids:
            continue
        q_obj = queries_by_id.get(qid, {'id': qid, 'vi_orig': ret.get('query', '')})

        # the retrieval scores are used for the document prior
        doc_scores = ret.get('scores')

        chunks = None
        for _attempt in range(5):
            try:
                chunks = selector.select_chunks(q_obj, ret['relevant_docs'],
                                       CHUNK_TOPK,
                                       doc_scores=doc_scores,
                                       dense=CHUNK_DENSE[0],
                                       dense_batch=CHUNK_BATCH[0])
                break
            except Exception as exc:
                oom = 'out of memory' in str(exc).lower()
                CHUNK_OOM_COUNT[0] += 1 if oom else 0
                print('  !! query {} failed ({}): {}'.format(
                    qid, 'CUDA OOM' if oom else 'ERROR', exc))
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                if not oom:
                    break
                # halve the batch and try again (the smaller batch sticks)
                CHUNK_BATCH[0] = max(1, CHUNK_BATCH[0] // 2)
                print('     -> the chunk batch is now {}'.format(CHUNK_BATCH[0]))
                if CHUNK_BATCH[0] == 1:
                    # even one text does not fit -> give the dense part up
                    print('     -> the dense scoring is given up for this stage')
                    CHUNK_DENSE[0] = False
                    break

        if chunks is None and CHUNK_DENSE[0]:
            # the last resort: no dense scoring at all for this query
            print('  !! query {} -> BM25 only (no dense scoring)'.format(qid))
            CHUNK_DENSE[0] = False
            try:
                chunks = selector.select_chunks(q_obj, ret['relevant_docs'],
                                       CHUNK_TOPK, doc_scores=doc_scores,
                                       dense=False)
            except Exception as exc:
                print('  !! query {} failed completely: {}'.format(qid, exc))
                chunks = []
        elif chunks is None:
            # the dense scoring is already off and it still failed
            try:
                chunks = selector.select_chunks(q_obj, ret['relevant_docs'],
                                       CHUNK_TOPK, doc_scores=doc_scores,
                                       dense=False)
            except Exception as exc:
                print('  !! query {} failed completely: {}'.format(qid, exc))
                chunks = []

        # A query must never be lost: if everything failed the query is kept
        # with an EMPTY chunk list (the documents are still submitted, so the
        # document half of the score is preserved).
        if chunks is None:
            print('  !! query {} -> no chunks at all (the documents are kept)'.format(qid))
            chunks = []

        # the quality guard of the submission: a chunk text must appear only once
        seen_texts = set()
        uniq_chunks = []
        for c in chunks:
            t = c['chunk_text']
            if t in seen_texts:
                dup_total += 1
                continue
            seen_texts.add(t)
            uniq_chunks.append(c)
        chunks = uniq_chunks

        try:
            qid_int = int(qid)
        except (TypeError, ValueError):
            qid_int = ret['id']
        row = {
            'id': qid_int,
            'relevant_docs': [str(d) for d in ret['relevant_docs']],
            'relevant_chunks': [{'doc_id': str(c['doc_id']), 'chunk_text': str(c['chunk_text'])}
                                for c in chunks],
        }
        submission.append(row)
        pending.append(row)
        chunk_counts.append(len(chunks))
        done_qids.add(qid)
        if len(pending) >= CHECKPOINT_EVERY:
            append_jsonl(CHUNKS_JSONL, pending)
            pending = []
            checkpoint.set_stage('chunk', status='IN_PROGRESS', done=len(submission),
                      total=len(retrieval_results), file='chunks.jsonl')
            # release the cached blocks of the GPU so the memory cannot creep up
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    append_jsonl(CHUNKS_JSONL, pending)
    CHUNK_JSON.write_text(json.dumps(submission, ensure_ascii=False), encoding='utf-8')
    checkpoint.set_stage('chunk', status='DONE', done=len(submission),
              total=len(retrieval_results), file='chunks.jsonl', json='chunk_checkpoint.json')
    print('chunk selection done:', len(submission), 'queries')
    print('  CUDA OOMs handled: {} | final batch: {} | dense scoring: {}'.format(
        CHUNK_OOM_COUNT[0], CHUNK_BATCH[0], 'on' if CHUNK_DENSE[0] else 'OFF (BM25 only)'))

    # --- the report of the chunk budget ---------------------------------------- #
    if chunk_counts:
        arr = np.asarray(chunk_counts, dtype=float)
        print()
        print('adaptive chunk count over the {} new queries:'.format(len(arr)))
        print('  mean {:.1f} | median {:.0f} | min {:.0f} | max {:.0f}'.format(
            arr.mean(), float(np.median(arr)), arr.min(), arr.max()))
        print('  percentiles p10/p25/p50/p75/p90 = {}'.format(
            [int(np.percentile(arr, p)) for p in (10, 25, 50, 75, 90)]))
        print('  duplicates removed: {}'.format(dup_total))

    return submission
