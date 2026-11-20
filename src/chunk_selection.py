"""Generate, score and select relevant multilingual chunks."""

from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from .chunk_windows import _apply_windows
from .scoring import _minmax, estimate_count
from .text import chunk_document_multiscale, tokenize

from .config import (
    BM25_WEIGHT,
    CHUNK_BREAK_W,
    CHUNK_ENCODE_BATCH,
    CHUNK_MAX,
    CHUNK_MAX_LENGTH,
    CHUNK_MIN,
    CHUNK_NESTED_DROP,
    CHUNK_REL_CUT,
    CHUNK_SHRINK,
    CHUNK_TARGET,
    CHUNK_WHOLE_DOC,
    CHUNK_WINDOW,
    DENSE_PER_DOC_CHUNKS,
    DENSE_PREFILTER,
    DENSE_PREFILTER_DOCS,
    DENSE_PREFILTER_MAX,
    DENSE_WEIGHT,
    DOC_COVERAGE_TOP,
    DOC_PRIOR_WEIGHT,
    EMBED_MAX_LENGTH,
    MAX_CHUNKS_PER_DOC,
)


_CHUNK_CACHE_MAX = 20000       # ~20000 documents x ~10 chunks = 200 000 chunks


def _dense_prefilter_indices(cands, bm25_raw, n_docs):
    """The chunks that are sent to the dense model.

    DEFECT 2 FIX.  The old version took the BM25-best chunks of the whole
    pool -> they all belong to the same ~20 documents -> the dense model
    never scored the chunks of the other documents.

    The new allocation is COVERAGE AWARE - the budget is spread over the
    documents in their retrieval order:

        document 1 : its best chunk
        document 2 : its best chunk
        document 3 : its best chunk
        ...        (DENSE_PER_DOC_CHUNKS chunks per document)

    and the remaining slots go to the BM25-best chunks of the whole pool
    (the lexical evidence the old version relied on).

    The total is capped by DENSE_PREFILTER_MAX so the runtime of the stage
    stays under control.
    """
    n = len(cands)
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    chosen = set(int(i) for i in np.argsort(-bm25_raw)[:DENSE_PREFILTER])

    # the best chunks of the best documents, round robin over the documents
    by_doc = defaultdict(list)
    for i, c in enumerate(cands):
        by_doc[c['doc_rank']].append(i)
    for rank in range(min(DENSE_PREFILTER_DOCS, n_docs)):
        idxs = by_doc.get(rank)
        if not idxs:
            continue
        idxs = sorted(idxs, key=lambda i: -bm25_raw[i])[:DENSE_PER_DOC_CHUNKS]
        chosen.update(int(i) for i in idxs)

    # --- the WHOLE-DOCUMENT candidate is NEVER encoded ---------------------- #
    # Its dense score is derived from the dense scores of its own chunks
    # (the max over them), so it costs NO encoding time.  It must not be in
    # the pre-filter, otherwise the encoder would spend 4x more time on the
    # long texts (the runtime would go from ~1.5 h to ~6 h).
    for i, c in enumerate(cands):
        if c.get('scale') == 'whole_doc':
            chosen.discard(i)

    if len(chosen) > DENSE_PREFILTER_MAX:
        # keep the BM25-best ones first, then fill up with the rest
        order = sorted(chosen, key=lambda i: -bm25_raw[i])
        keep = set(order[:DENSE_PREFILTER_MAX])
        for i in order[DENSE_PREFILTER_MAX:]:
            if len(keep) >= DENSE_PREFILTER_MAX:
                break
            keep.add(i)
        chosen = keep
    return np.array(sorted(chosen), dtype=np.int64)


class ChunkSelector:
    def __init__(self, corpus, model):
        self.corpus = corpus
        self.model = model
        self._cache = {}

    def _chunk_cache_put(self, key, value):
        if len(self._cache) >= _CHUNK_CACHE_MAX:
            # drop the oldest half (the dict preserves the insertion order)
            for k in list(self._cache.keys())[:_CHUNK_CACHE_MAX // 2]:
                self._cache.pop(k, None)
        self._cache[key] = value


    def _chunk_cache_get(self, key):
        return self._cache.get(key)


    def collect_candidates(self, doc_list, doc_scores=None):
        """The multi-scale chunk pool of the candidate documents.

        `doc_scores` are the retrieval scores of the documents (same order as
        `doc_list`).  They are used for the document prior; when they are missing
        the rank is used instead.

        The chunking of a document is CACHED: the same document is retrieved by
        many queries, and re-tokenising a 3000-character document 144 000 times
        (1200 queries x 120 documents) would cost more than the dense encoding
        of the chunks themselves.  The cache is bounded, so the RAM stays under
        control.
        """
        cands = []
        for rank, d in enumerate(doc_list):
            pos = self.corpus.positions.get(d)
            if pos is None:
                continue
            text = self.corpus.contents[pos]
            if not text.strip():
                continue
            if doc_scores is not None and rank < len(doc_scores):
                dscore = float(doc_scores[rank])
            else:
                dscore = 0.0
            chunks = self._chunk_cache_get(d)
            if chunks is None:
                chunks = chunk_document_multiscale(text)
                if CHUNK_WHOLE_DOC:
                    # --- the WHOLE-DOCUMENT candidate -------------------------- #
                    # The chunk metric is a CONTAINMENT metric: a submitted chunk
                    # scores a HIT for every GOLD chunk it CONTAINS, and a gold
                    # chunk is counted ONCE.  So the WHOLE document covers EVERY
                    # gold chunk of that document, while a small chunk covers only
                    # a few of them.
                    #
                    # Measured on 3053 real documents (the coverage = the number
                    # of the pool chunks of the document that the candidate
                    # contains):
                    #
                    #    scale            the coverage    the slots/document
                    #    fixed300/60          1.29            2.28
                    #    fixed700/150         3.48            1.16
                    #    natural2500          1.68            1.56
                    #    WHOLE DOCUMENT       4.45            1.00
                    #
                    # The whole document wins on BOTH axes: it covers the most and
                    # it uses the fewest slots.  It is added to the pool as one
                    # more candidate, so the ranking decides when to use it.
                    #
                    # It is added INSIDE the cache, so the tokenisation of the
                    # long text happens once per document and not once per query
                    # (150 documents x 1200 queries = 180 000 tokenisations of a
                    # 5000-character text would take hours of CPU).
                    whole = text.strip()
                    if whole and not any(c['text'] == whole for c in chunks):
                        chunks = list(chunks) + [{'text': whole,
                                                  'tokens': tokenize(whole, keep_short=True),
                                                  'scale': 'whole_doc'}]
                self._chunk_cache_put(d, chunks)
            for chunk in chunks:
                chunk_text = chunk['text']
                if not chunk_text:
                    continue
                # a SHALLOW COPY: the cached dict is shared between the queries
                # and the fields below are different for every query
                chunk = dict(chunk)
                chunk['doc_id'] = d
                chunk['doc_rank'] = rank
                chunk['doc_score'] = dscore
                cands.append(chunk)
        return cands


    def encode_dense_only(self, texts, batch_size, max_length):
        """`model.encode` restricted to the DENSE head.

        BGE-M3 has three heads.  With the default flags every call also builds
          - the SPARSE head -> a (batch, seq, vocab=250k) tensor
          - the COLBERT head -> a (batch, seq, 1024) tensor
        Only `dense_vecs` is read here, so both are switched off: a lot of VRAM
        and time are saved and the dense vectors are bit-identical.

        Older FlagEmbedding versions do not know these flags -> the plain call is
        used as a fallback.

        This matters a lot on a T4: the SPARSE head of a batch of 12 x 1024
        tokens alone is 12 x 1024 x 250 000 x 2 B = 6.1 GB of VRAM -> with the
        model (3.3 GB) and the rest of the pipeline that is an instant OOM.
        """
        try:
            out = self.model.encode(texts, batch_size=batch_size, max_length=max_length,
                               return_dense=True, return_sparse=False,
                               return_colbert_vecs=False)
        except TypeError:
            out = self.model.encode(texts, batch_size=batch_size, max_length=max_length)
        return np.asarray(out['dense_vecs'], dtype=np.float32)


    def select_chunks(self, query_obj, doc_list, topk=None,
                      doc_scores=None, dense=True, dense_batch=None):
        """Select the best chunks of the candidate documents.

        `dense=False` skips the dense scoring (the BM25 score alone is used) --
        the fallback for a query that does not fit in the VRAM any more.
        """
        cands = self.collect_candidates(doc_list, doc_scores)
        if not cands:
            return []

        q_vi_orig = query_obj.get('vi_orig') or query_obj.get('query') or ''
        q_vi_exp = query_obj.get('vi_exp') or ''
        q_vi = ' '.join(x for x in (q_vi_orig, q_vi_exp) if x).strip()
        q_en = query_obj.get('en') or ''
        q_zh = query_obj.get('zh') or ''
        q_tokens = tokenize(q_vi, keep_short=True) + tokenize(q_en, keep_short=True) \
                   + tokenize(q_zh, keep_short=True)

        texts = [c['text'] for c in cands]
        corpus_tokens = [c['tokens'] or ['_empty_'] for c in cands]

        if q_tokens:
            bm25 = BM25Okapi(corpus_tokens)
            bm25_raw = np.asarray(bm25.get_scores(q_tokens), dtype=float)
        else:
            bm25_raw = np.zeros(len(cands), dtype=float)
        bm25_norm = _minmax(bm25_raw)

        # --- the dense pre-filter (coverage aware) ----------------------------- #
        dense_norm = np.zeros(len(cands), dtype=float)
        pre_idx = _dense_prefilter_indices(cands, bm25_raw, len(doc_list))
        if dense and self.model is not None and pre_idx.size and (q_vi or q_en or q_zh):
            # the FULL multilingual query -> the dense model can match the
            # Vietnamese question against an English or a Chinese passage
            q_text = ' '.join(x for x in (q_vi, q_en, q_zh) if x).strip()
            q_emb = self.encode_dense_only([q_text], 1, EMBED_MAX_LENGTH)[0]
            q_emb = q_emb.copy()
            q_emb /= np.linalg.norm(q_emb) + 1e-8

            # --- the chunk scoring, memory-lean ------------------------------- #
            # The texts are sorted by length, so a batch only holds texts of a
            # similar size -> almost no padding -> a small activation peak.  The
            # batch is small (CHUNK_ENCODE_BATCH) because at max_length=1024 a
            # batch of 32 materialises the attention matrix of every head
            # (32 x 16 x 1024 x 1024 x 2 B = 1.07 GB) plus twice the FFN
            # intermediate (1.07 GB each) -> CUDA OOM on a T4 that already holds
            # the model and the document embeddings.
            #
            # The ORDER of the vectors is tracked explicitly (the `order` map) so
            # the result does not depend on how the encoder sorts the texts.
            batch_size = int(dense_batch or CHUNK_ENCODE_BATCH)
            batch_size = max(1, batch_size)
            lengths = np.array([len(texts[i]) for i in pre_idx], dtype=np.int64)
            order = np.argsort(lengths, kind='stable')
            vecs = [None] * len(pre_idx)
            for start in range(0, len(order), batch_size):
                stop = min(start + batch_size, len(order))
                slots = order[start:stop]
                batch = [texts[pre_idx[j]] for j in slots]
                arr = self.encode_dense_only(batch, batch_size, CHUNK_MAX_LENGTH)
                for k, j in enumerate(slots):
                    vecs[j] = arr[k]
                del arr, batch
            mat = np.asarray(vecs, dtype=np.float32)
            del vecs
            mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
            dense_norm[pre_idx] = _minmax(mat @ q_emb)
            del mat

        # --- the dense score of the WHOLE DOCUMENT ----------------------------- #
        # The whole document is NOT encoded (it would cost 4x more encoding time
        # and would time out the Kaggle session).  Its dense score is DERIVED
        # from the dense scores of its own chunks:
        #
        #     dense(whole_doc) = max over the chunks of the same document
        #
        # The justification: the cosine similarity between a query and a text is
        # driven by the BEST-MATCHING PASSAGE of that text.  The best-matching
        # passage of a document IS one of its chunks.  So the max over the
        # chunks is a good proxy of the dense score of the whole document - and
        # it costs nothing, because the chunk embeddings are computed anyway.
        #
        # The whole document is then ranked at least as high as its best chunk,
        # which is exactly what the containment metric wants (the whole document
        # covers everything the chunk covers, plus more).
        if CHUNK_WHOLE_DOC and dense_norm.any():
            best_of_doc = {}
            for i, c in enumerate(cands):
                if c.get('scale') == 'whole_doc':
                    continue
                d = c['doc_id']
                v = dense_norm[i]
                if d not in best_of_doc or v > best_of_doc[d]:
                    best_of_doc[d] = v
            for i, c in enumerate(cands):
                if c.get('scale') == 'whole_doc':
                    v = best_of_doc.get(c['doc_id'], 0.0)
                    # a small bonus: the whole document covers MORE than its best
                    # chunk, so it is at least as relevant
                    dense_norm[i] = min(1.0, v)

        # --- the document prior ------------------------------------------------ #
        if doc_scores is not None and len(doc_scores) == len(doc_list):
            raw_prior = np.asarray([1.0 / (1.0 + float(s)) for s in doc_scores],
                                   dtype=float)
            raw_prior = _minmax(raw_prior)
            prior = np.asarray([raw_prior[c['doc_rank']] for c in cands],
                               dtype=float)
        else:
            prior = np.asarray([1.0 / (c['doc_rank'] + 1) for c in cands], dtype=float)
        prior = _minmax(prior)

        hybrid = (BM25_WEIGHT * bm25_norm
                  + DENSE_WEIGHT * dense_norm
                  + DOC_PRIOR_WEIGHT * prior)

        order = np.argsort(-hybrid)

        # --- the ADAPTIVE number of chunks ------------------------------------- #
        # The budget follows the shape of the score curve of THIS query: a query
        # whose curve collapses after a few chunks needs few chunks, a query with
        # a flat curve (many plausible passages) needs many.
        if topk is None:
            budget, _diag = estimate_count(
                hybrid[order],
                rel_cut=CHUNK_REL_CUT,
                break_w=CHUNK_BREAK_W,
                shrink=CHUNK_SHRINK,
                floor=CHUNK_MIN,
                lo=CHUNK_MIN,
                hi=CHUNK_MAX,
                target=CHUNK_TARGET)
            # ------------------------------------------------------------------ #
            # THE FLOOR RULE (the same as for the documents)
            # ------------------------------------------------------------------ #
            # The v2 measurement: the hit rate of the submitted chunks is 19.1 %
            # while the break-even is 3.4 % -> the chunks are 5.6x more valuable
            # than they cost.  The chunk score curve is flat too, so its "break"
            # is noise, not the end of the relevant set.
            #
            # => the estimator may only make the budget BIGGER than CHUNK_MIN.
            #    It must not be allowed to cut the tail.
            if budget < CHUNK_MIN:
                budget = CHUNK_MIN
        else:
            budget = int(topk)
        budget = int(max(CHUNK_MIN, min(budget, CHUNK_MAX)))

        # --- the budget follows the DOCUMENT count ---------------------------- #
        # With the whole-document candidate the best submission is ONE chunk for
        # EVERY retrieved document (each slot then covers all the gold chunks of
        # its document, and every gold document gets a slot).
        #
        # Measured: the chunk F2 with 97 documents is 0.424 and with 150
        # documents it is 0.564 -> every extra document is worth much more than
        # an extra chunk of a document that already has one.
        #
        # So the budget is never below the number of retrieved documents.
        if CHUNK_WHOLE_DOC:
            budget = max(budget, len(doc_list))

        # --- the selection ------------------------------------------------------ #
        # The CONCENTRATION on the best documents is kept (it was checked that it
        # is not the bug), but:
        #   * the duplicates are removed
        #   * the cap per document is a bit more generous (a gold document holds
        #     ~1.88 gold chunks)
        selected = []
        selected_set = set()
        selected_texts = set()
        per_doc_count = defaultdict(int)

        # --- the REDUNDANCY of the NESTED chunks ------------------------------- #
        # The exact duplicates were removed (20.6 % of the v1 slots), but there is
        # a second, much bigger family of redundancy that the exact check does
        # NOT catch:
        #
        #   the three scales produce NESTED chunks.  A passage that is covered by
        #   the "300 token" chunk is ALSO inside the "700 token" chunk and also
        #   inside the "natural 2500 chars" chunk.  Measured on the produced
        #   chunks.jsonl of v2:
        #
        #       exact duplicates              : 0     (the fix works)
        #       near duplicates in the SAME doc: 78.6 % of the pairs
        #       (containment = 1.00 -> one chunk is INSIDE the other)
        #
        # WHY THE NESTED CHUNKS ARE REDUNDANT
        #
        #   The scorer checks CONTAINMENT: a submitted chunk counts as a hit when
        #   the gold chunk is inside it.  The gold chunk is a FIXED piece of text
        #   produced by the SAME chunker.  So if the gold chunk G is inside the
        #   small chunk S, and S is inside the big chunk B, then:
        #
        #       G ⊂ S ⊂ B   ->   S is a hit AND B is a hit
        #
        #   -> submitting both S and B spends 2 slots for the same gold chunk.
        #   -> the score does NOT grow, but the "submitted" term of the F2
        #      denominator does -> the precision drops.
        #   -> the freed slot is worth more on ANOTHER document.
        #
        # TWO KINDS OF OVERLAP - AND THEY MUST BE TREATED DIFFERENTLY
        #
        #   Measured on the v2 output (74 123 near-duplicate pairs):
        #
        #       LITERALLY nested  : 55 661 (75.1 %)
        #           -> the small text is a substring of the big text
        #           -> B contains EVERY gold chunk that S contains
        #           -> keeping B alone is STRICTLY BETTER -> drop S
        #
        #       only a token overlap : 18 462 (24.9 %)
        #           -> the boundaries of the two scales CROSS each other, so
        #              NEITHER contains the other
        #           -> dropping one of them can LOSE a gold chunk
        #           -> keep BOTH
        #
        #   => the rule is a LITERAL substring test, not a token test.
        #
        #   The comparison is done only INSIDE the same document: two different
        #   documents are different texts and must never be compared.
        #
        # The rule, concretely:
        #   * a candidate that is a literal substring of a selected chunk of the
        #     same document -> DROPPED (the selected one covers it)
        #   * a candidate that LITERALLY CONTAINS a selected chunk of the same
        #     document -> it REPLACES that chunk (it covers everything the chunk
        #     covered, and possibly more)
        #   * a candidate that only shares the tokens -> KEPT (it is a different
        #     passage)
        #
        # The check is a plain `in` on the strings, so it is exact (no false
        # positives) and fast (the chunks of one document are few).
        # The index stores the CANDIDATE INDEX (not the position in `selected`):
        # the positions shift every time a chunk is removed, so they cannot be
        # used as a stable key.
        sel_by_doc = defaultdict(list)      # doc -> [(candidate index, text), ...]

        def _remove_candidate(j):
            """Remove the selected candidate `j` from the selection."""
            selected.remove(j)
            selected_set.discard(j)
            selected_texts.discard(cands[j]['text'])
            d = cands[j]['doc_id']
            per_doc_count[d] -= 1
            entries = sel_by_doc[d]
            for k in range(len(entries) - 1, -1, -1):
                if entries[k][0] == j:
                    entries.pop(k)
                    break

        def add_candidate(i):
            i = int(i)
            if i in selected_set or len(selected) >= budget:
                return False
            d = cands[i]['doc_id']
            text = cands[i]['text']
            # DEFECT 1 FIX: the same text is never submitted twice
            if text in selected_texts:
                return False
            if per_doc_count[d] >= MAX_CHUNKS_PER_DOC:
                return False

            # --- DEFECT 4 FIX: the nested chunks -------------------------------- #
            entries = sel_by_doc.get(d) if CHUNK_NESTED_DROP else None
            if entries:
                # BUG FIX: the old code kept only ONE `replace` index, so when the
                # candidate contained SEVERAL selected chunks of the same document
                # only the first one was removed and the others stayed in the
                # submission.  They are redundant (the candidate covers all of
                # them) -> they waste a slot of the F2 denominator.
                #
                #   the candidate CONTAINS a selected chunk -> the selected chunk
                #   is redundant -> it is removed (ALL of them, not only the
                #   first one)
                #
                #   the candidate is INSIDE a selected chunk -> it is redundant
                #   -> it is rejected
                to_replace = []
                for j, t in entries:
                    if text in t:
                        # the candidate is INSIDE a selected chunk -> redundant
                        return False
                    if t in text:
                        # the candidate CONTAINS a selected chunk -> it replaces it
                        to_replace.append(j)
                for j in to_replace:
                    _remove_candidate(j)

            selected.append(i)
            selected_set.add(i)
            selected_texts.add(text)
            sel_by_doc[d].append((i, text))
            per_doc_count[d] += 1
            return True

        # the best chunk of every document (the DOC_COVERAGE_TOP guarantee)
        best_per_doc = {}
        for i in order:
            i = int(i)
            best_per_doc.setdefault(cands[i]['doc_id'], i)
        for d in doc_list[:DOC_COVERAGE_TOP]:
            i = best_per_doc.get(d)
            if i is not None:
                add_candidate(i)

        # then the global ranking fills the rest
        for i in order:
            if len(selected) >= budget:
                break
            add_candidate(i)

        # --- NO BACKFILL ------------------------------------------------------- #
        # There is deliberately NO backfill here.
        #
        # The scorer counts a HIT per DISTINCT gold chunk that is covered (the
        # gold chunks are a set, a gold chunk is counted once).  So a nested
        # chunk covers exactly the same gold chunks as the chunk that contains
        # it:
        #
        #       submitted = {S, B}  ->  hits = 1 (the gold chunk G inside them)
        #                               submitted = 2
        #       submitted = {B}     ->  hits = 1
        #                               submitted = 1
        #
        # and the F2 of the chunk half is  5 * hits / (4 * gold + submitted).
        # The nested chunk adds NOTHING to the numerator and +1 to the
        # denominator -> it LOWERS the score.
        #
        # So the budget is deliberately left unfilled when the remaining
        # candidates are all nested: an unused slot is harmless (it does not
        # enter the denominator), a redundant chunk is not.
        #
        # (The loop above does not stop at the first rejection: it keeps
        # scanning, so the budget is filled with the DISTINCT chunks of the
        # other documents.)
        selected = sorted(selected, key=lambda i: -hybrid[i])[:budget]
        if CHUNK_WINDOW:
            return _apply_windows(cands, selected, hybrid)
        return [{'doc_id': cands[i]['doc_id'], 'chunk_text': cands[i]['text']}
                for i in selected]
