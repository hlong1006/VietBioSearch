"""Build or restore the shard embeddings and prepare the GPU matrices."""

import gc
import shutil
from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import (
    EMBED_BATCH_TOKENS,
    EMBED_MAX_LENGTH,
    EMBED_MAX_SEQS,
    EMBED_SUBSAMPLE,
)


@dataclass
class DocumentEmbeddings:
    matrix: np.ndarray
    gpu_by_language: dict
    language_row_by_position: np.ndarray | None


def load_document_embeddings(corpus, model, device, checkpoint):
    n_shards = len(corpus.shard_offsets) - 1
    emb_dim = 1024
    EMB_DIR = checkpoint.work_dir / 'emb_shards'
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    EMB_FULL = checkpoint.work_dir / 'doc_emb.npy'
    EMB_MARK = checkpoint.work_dir / 'doc_emb.done'


    def emb_shard_npy(i):
        return EMB_DIR / 'emb_{:04d}.npy'.format(i)


    def emb_shard_done(i):
        return EMB_DIR / 'emb_{:04d}.done'.format(i)


    def embed_texts(texts, out):
        """Encode `texts` into the pre-allocated (len, dim) fp16 matrix `out`."""
        lengths = np.array([len(t) for t in texts], dtype=np.int32)
        order = np.argsort(lengths)          # length sorted -> homogeneous batches
        batch, batch_tokens = [], 0

        def flush(idx_batch):
            if not idx_batch:
                return
            # only the DENSE head is needed -> the SPARSE head (batch x seq x
            # 250k vocab) and the COLBERT head are switched off.  They would
            # cost ~6 GB of VRAM for a batch of 12 x 1024 tokens and the dense
            # vectors are bit-identical.
            try:
                vecs = model.encode([texts[i] for i in idx_batch],
                                    batch_size=len(idx_batch),
                                    max_length=EMBED_MAX_LENGTH,
                                    return_dense=True, return_sparse=False,
                                    return_colbert_vecs=False)['dense_vecs']
            except TypeError:
                vecs = model.encode([texts[i] for i in idx_batch],
                                    batch_size=len(idx_batch),
                                    max_length=EMBED_MAX_LENGTH)['dense_vecs']
            mat = np.asarray(vecs, dtype=np.float32)
            mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
            out[idx_batch] = mat.astype(np.float16)

        for k in order:
            k = int(k)
            est = min(int(lengths[k]), EMBED_MAX_LENGTH * 4)
            if batch and (batch_tokens + est > EMBED_BATCH_TOKENS
                          or len(batch) >= EMBED_MAX_SEQS):
                flush(batch)
                batch, batch_tokens = [], 0
            batch.append(k)
            batch_tokens += est
        flush(batch)
        return out


    # --- 0) is the FULL matrix already available?  then nothing has to be done -- #
    full_emb_path = checkpoint.resolve_with_marker('doc_emb.npy', 'doc_emb.done')
    SKIP_EMBED = full_emb_path is not None

    todo = []
    if SKIP_EMBED:
        print('the full embedding matrix is already there ({}), '
              'the embedding stage is SKIPPED'.format(full_emb_path))
    else:
        for i in range(n_shards):
            if emb_shard_done(i).exists() and emb_shard_npy(i).exists():
                continue
            # maybe a previous session already computed it -> copy it over
            copied = False
            for base in checkpoint.input_dirs:
                src_npy = base / 'emb_shards' / 'emb_{:04d}.npy'.format(i)
                src_done = base / 'emb_shards' / 'emb_{:04d}.done'.format(i)
                if src_npy.exists() and src_done.exists():
                    try:
                        shutil.copy2(src_npy, emb_shard_npy(i))
                        shutil.copy2(src_done, emb_shard_done(i))
                        print('  restored emb shard {:04d} from the dataset'.format(i))
                        copied = True
                    except Exception as exc:
                        print('  could not restore emb shard {:04d}: {}'.format(i, exc))
                    break
            if not copied:
                todo.append(i)

    n_done = n_shards - len(todo)
    print('embedding shards: {}/{} done, {} to compute'.format(n_done, n_shards, len(todo)))
    checkpoint.set_stage('embed', status='DONE' if n_done == n_shards else 'IN_PROGRESS',
              shards_total=n_shards, shards_done=n_done,
              files='emb_shards/emb_%04d.npy' % n_done if n_done else 'none')

    # --- 1) compute the missing shards ---------------------------------------- #
    if not SKIP_EMBED:
        for i in tqdm(todo, desc='embedding shards'):
            lo, hi = int(corpus.shard_offsets[i]), int(corpus.shard_offsets[i + 1])
            out = np.zeros((hi - lo, emb_dim), dtype=np.float16)
            if EMBED_SUBSAMPLE:
                # keep the original behaviour: only the first EMBED_SUBSAMPLE docs
                # are embedded, the rest stays zero
                keep = max(0, min(hi - lo, EMBED_SUBSAMPLE - lo))
                if keep > 0:
                    embed_texts(corpus.contents[lo:lo + keep], out[:keep])
            else:
                embed_texts(corpus.contents[lo:hi], out)
            np.save(emb_shard_npy(i), out)
            emb_shard_done(i).write_text('ok', encoding='utf-8')
            del out
            gc.collect()
            torch.cuda.empty_cache()

        n_done = sum(1 for i in range(n_shards) if emb_shard_done(i).exists())
        checkpoint.set_stage('embed', status='DONE' if n_done == n_shards else 'IN_PROGRESS',
                  shards_total=n_shards, shards_done=n_done,
                  files='emb_shards/emb_%04d.npy' % n_done)

    # --- 2) assemble the shards into one memmap (only if not already complete) - #
    full = checkpoint.resolve_with_marker('doc_emb.npy', 'doc_emb.done')
    if full is None:
        print('assembling doc_emb.npy from the shard files ...')
        if EMB_FULL.exists():
            EMB_FULL.unlink()
        doc_emb = np.lib.format.open_memmap(EMB_FULL, mode='w+', dtype=np.float16,
                                             shape=(len(corpus.ids), emb_dim))
        for i in range(n_shards):
            part = np.load(emb_shard_npy(i), mmap_mode='r')
            lo = int(corpus.shard_offsets[i])
            doc_emb[lo:lo + len(part)] = part[:]
            del part
        doc_emb.flush()
        EMB_MARK.write_text('ok', encoding='utf-8')
        print('assembled', doc_emb.shape)
    else:
        doc_emb = np.load(full, mmap_mode='r')
        print('reloaded embeddings', doc_emb.shape, 'from', full)

    # --- 3) per-language CONTIGUOUS embedding matrices on the GPU -------------- #
    #
    # WHY per language and not one big matrix:
    #   the old dense_candidates() evaluated
    #       doc_emb_gpu[lang_idx_gpu[lang]].float() @ q
    #   which (a) copied the WHOLE language out of the big matrix and up-cast it
    #   to fp32 -> 714.891 x 1024 x 4 B = 2.73 GB of VRAM per query and (b) ran
    #   while the BGE-M3 model was still resident on the same 14 GB T4 ->
    #   OutOfMemoryError on the very first query of cell 11.
    #
    #   Now every language is stored CONTIGUOUSLY in fp16 and the similarity is a
    #   single cuBLAS GEMM (fp16 inputs, fp32 accumulation):
    #       (n_lang, 1024) fp16  @  (1024,) fp16  ->  (n_lang,) fp16
    #   -> no copy at all, 2.26 GB for the three languages together, ~10 ms/query.
    #
    # The big global matrix is dropped afterwards: it would cost another 2.26 GB
    # of VRAM for nothing.

    lang_emb_gpu = {}          # lang -> (n_lang, dim) fp16 tensor on the GPU
    local_of_pos = None        # global position -> row inside the language block

    try:

        # global position -> local row inside the block of its language
        local_of_pos = np.zeros(len(corpus.ids), dtype=np.int32)
        for _lang, _idx in corpus.language_indices.items():
            _idx_arr = np.asarray(_idx, dtype=np.int64)
            local_of_pos[_idx_arr] = np.arange(len(_idx_arr), dtype=np.int32)

        _GATHER_STEP = 65536
        for _lang, _idx in corpus.language_indices.items():
            _idx_arr = np.asarray(_idx, dtype=np.int64)
            try:
                _host = np.empty((len(_idx_arr), emb_dim), dtype=np.float16)
                for _s in range(0, len(_idx_arr), _GATHER_STEP):
                    _e = min(_s + _GATHER_STEP, len(_idx_arr))
                    _host[_s:_e] = doc_emb[_idx_arr[_s:_e]]
                lang_emb_gpu[_lang] = torch.from_numpy(_host).to(device)
                del _host
                gc.collect()
                torch.cuda.empty_cache()
                print('  gpu {:<3} {:>9,} rows x {} dims  ({:.2f} GB)'.format(
                    _lang, len(_idx_arr), emb_dim,
                    len(_idx_arr) * emb_dim * 2 / 1024 ** 3))
            except Exception as exc:
                # one language that does not fit must not kill the others
                print('  could not put {} on the GPU ({}), it stays on the CPU'
                      .format(_lang, exc))
                lang_emb_gpu.pop(_lang, None)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        print('embeddings on GPU (per language):',
              {k: tuple(v.shape) for k, v in lang_emb_gpu.items()})
    except Exception as exc:
        print('could not move the embeddings to the GPU:', exc)
        lang_emb_gpu = {}
        local_of_pos = None


    try:
        torch.cuda.empty_cache()
        print('VRAM allocated: {:.2f} GB | reserved: {:.2f} GB'.format(
            torch.cuda.memory_allocated() / 1024 ** 3,
            torch.cuda.memory_reserved() / 1024 ** 3))
    except Exception:
        pass

    return DocumentEmbeddings(doc_emb, lang_emb_gpu, local_of_pos)
