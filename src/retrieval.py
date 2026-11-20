"""Hybrid multilingual document retrieval and ranking."""

from collections import defaultdict

import numpy as np

from .scoring import _minmax, _topk_positions, estimate_count, rrf_fusion
from .text import tokenize

from .config import (
    ADAPTIVE_DOCS,
    ADAPTIVE_DOC_AGREE,
    ADAPTIVE_DOC_BREAK_W,
    ADAPTIVE_DOC_MAX,
    ADAPTIVE_DOC_MIN,
    ADAPTIVE_DOC_REL_CUT,
    ADAPTIVE_DOC_SHRINK,
    ADAPTIVE_DOC_TARGET,
    ADAPTIVE_POOL,
    ADAPTIVE_W_BM25,
    ADAPTIVE_W_DENSE,
    ADAPTIVE_W_RRF,
    BM25_TOPK,
    DENSE_CHUNK,
    DENSE_TOPK,
    EMBED_MAX_LENGTH,
    FINAL_TOPK_DOCS,
    LANG_WEIGHTS,
    RERANK_TOPK,
)


class DocumentRetriever:
    def __init__(self, corpus, bm25_indexes, embeddings, model, device):
        self.corpus = corpus
        self.bm25_indexes = bm25_indexes
        self.embeddings = embeddings
        self.model = model
        self.device = device

    def bm25_search(self, query_text, lang, topk=BM25_TOPK):
        retriever = self.bm25_indexes.get(lang)
        idx_list = self.corpus.language_indices.get(lang, [])
        if retriever is None or not idx_list or not query_text:
            return []
        q_tokens = tokenize(query_text, keep_short=True)
        if not q_tokens:
            return []
        k = min(topk, len(idx_list))
        try:
            res = retriever.retrieve([q_tokens], k=k, show_progress=False)
        except Exception as exc:
            print('BM25 retrieve failed:', exc)
            return []
        out = []
        for local_id, score in zip(res.documents[0], res.scores[0]):
            if score > 0:
                out.append((idx_list[int(local_id)], float(score)))
        return out


    def _query_vector(self, query_text):
        # only the DENSE head -> the SPARSE head and the COLBERT head are
        # switched off (they would cost a lot of VRAM for nothing)
        try:
            q_emb = self.model.encode([query_text], batch_size=1,
                                 max_length=EMBED_MAX_LENGTH,
                                 return_dense=True, return_sparse=False,
                                 return_colbert_vecs=False)['dense_vecs'][0]
        except TypeError:
            q_emb = self.model.encode([query_text], batch_size=1,
                                 max_length=EMBED_MAX_LENGTH)['dense_vecs'][0]
        q = np.asarray(q_emb, dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-8
        return q


    def dense_candidates(self, query_text, lang, topk=DENSE_TOPK):
        """Top-k documents of one language by cosine similarity.

        GPU path: ONE fp16 GEMM against the contiguous per-language block -> no
        copy of the language matrix, so no OOM.  If the GPU cannot be used the
        blocked numpy path below takes over.
        """
        idx_list = self.corpus.language_indices.get(lang, [])
        if not idx_list or not query_text:
            return []
        q = self._query_vector(query_text)

        mat = self.embeddings.gpu_by_language.get(lang)
        if mat is not None:
            try:
                import torch
                q32 = torch.from_numpy(q).to(self.device)
                n = mat.shape[0]
                scores = torch.empty(n, dtype=torch.float32, device=self.device)
                # CHUNKED up-cast: the old code did
                #     doc_emb_gpu[lang_idx_gpu[lang]].float() @ q
                # i.e. it materialised the WHOLE language as fp32 (2.73 GB for
                # Vietnamese) -> CUDA OOM.  Here the up-cast happens in blocks of
                # DENSE_CHUNK rows, so the peak is ~DENSE_CHUNK * 1024 x 4 B
                # (134 MB for 32768 rows) and the arithmetic is IDENTICAL to the
                # old code (fp16 -> fp32 is exact, no renormalisation, same dot
                # product) -> the ranking does not change at all.
                for s in range(0, n, DENSE_CHUNK):
                    e = min(s + DENSE_CHUNK, n)
                    scores[s:e] = mat[s:e].float() @ q32
                sims = scores.cpu().numpy()
                out = []
                for t in _topk_positions(sims, topk):
                    s = float(sims[t])
                    if s > 0:
                        out.append((idx_list[int(t)], s))
                return out
            except Exception as exc:
                print('dense_candidates on the GPU failed ({}), '
                      'falling back to the CPU'.format(exc))
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        # CPU fallback: blocked so the RAM peak stays small.
        # NO renormalisation here either -> identical to the GPU path and to the
        # old implementation (the stored embeddings are already L2-normalised).
        idx_arr = np.asarray(idx_list, dtype=np.int64)
        scores = np.empty(len(idx_arr), dtype=np.float32)
        for s in range(0, len(idx_arr), DENSE_CHUNK):
            e = min(s + DENSE_CHUNK, len(idx_arr))
            scores[s:e] = np.asarray(self.embeddings.matrix[idx_arr[s:e]],
                                     dtype=np.float32) @ q
        out = []
        for t in _topk_positions(scores, topk):
            v = float(scores[t])
            if v > 0:
                out.append((idx_list[int(t)], v))
        return out


    def dense_rerank(self, query_text, candidate_indices, topk=RERANK_TOPK, lang=None):
        """Re-rank the candidates of ONE language by cosine similarity."""
        if not candidate_indices or not query_text:
            return []
        q = self._query_vector(query_text)

        scores = None
        mat = self.embeddings.gpu_by_language.get(lang) if lang else None
        if mat is not None and self.embeddings.language_row_by_position is not None:
            try:
                import torch
                cand = np.asarray(candidate_indices, dtype=np.int64)
                loc = torch.from_numpy(self.embeddings.language_row_by_position[cand].astype(np.int64)).to(self.device)
                rows = mat.index_select(0, loc).float()
                scores = (rows @ torch.from_numpy(q).to(self.device)).cpu().numpy()
            except Exception as exc:
                print('dense rerank on the GPU failed ({}), '
                      'falling back to the CPU'.format(exc))
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                scores = None

        if scores is None:
            # the stored embeddings are already L2-normalised -> no renormalisation
            rows = np.asarray(self.embeddings.matrix[list(candidate_indices)], dtype=np.float32)
            scores = rows @ q

        ranked = sorted(zip(candidate_indices, scores), key=lambda x: x[1], reverse=True)
        return [(int(i), float(s)) for i, s in ranked[:topk]]


    def retrieve_documents(self, query_obj):
        """-> (doc_ids, scores, info) for one query.

        `doc_ids` is the final list of documents (its LENGTH is the adaptive
        part), `scores` the hybrid score of every returned document (the order
        of the documents is the proven RRF order).
        """
        q_vi_orig = query_obj.get('vi_orig') or query_obj.get('query') or ''
        q_vi_exp = query_obj.get('vi_exp') or ''
        # Query expansion improves lexical recall while retaining the original
        # wording as the first part of the query.
        q_vi = ' '.join(x for x in (q_vi_orig, q_vi_exp) if x).strip()
        q_en = query_obj.get('en') or ''
        q_zh = query_obj.get('zh') or ''

        per_lang = {}
        bm25_best = {}
        dense_best = {}
        lang_membership = defaultdict(set)

        for lang, qtext in (('vi', q_vi), ('en', q_en), ('zh', q_zh)):
            if not qtext:
                per_lang[lang] = []
                continue
            cands = self.bm25_search(qtext, lang, BM25_TOPK)
            dense = self.dense_candidates(qtext, lang, DENSE_TOPK)
            seen = set()
            for p, s in cands:
                bm25_best[p] = max(bm25_best.get(p, 0.0), float(s))
                seen.add(p)
            for p, s in dense:
                dense_best[p] = max(dense_best.get(p, 0.0), float(s))
                if p not in seen:
                    cands.append((p, s))
                    seen.add(p)
            per_lang[lang] = self.dense_rerank(qtext, [c[0] for c in cands],
                                          RERANK_TOPK, lang=lang)
            for p, _s in per_lang[lang]:
                lang_membership[p].add(lang)

        weights = [LANG_WEIGHTS['vi'], LANG_WEIGHTS['en'], LANG_WEIGHTS['zh']]
        fused = rrf_fusion([per_lang['vi'], per_lang['en'], per_lang['zh']], weights)
        if not fused:
            return [], [], {}

        # the RRF ORDER is the proven one and is NOT changed
        pool = fused[:ADAPTIVE_POOL]
        positions = [p for p, _ in pool]

        # the hybrid score is only used to estimate the adaptive cut
        rrf_scores = np.asarray([s for _, s in pool], dtype=float)
        dsc = np.asarray([dense_best.get(p, 0.0) for p in positions], dtype=float)
        bsc = np.asarray([bm25_best.get(p, 0.0) for p in positions], dtype=float)
        curve = (ADAPTIVE_W_DENSE * _minmax(dsc)
                 + ADAPTIVE_W_BM25 * _minmax(bsc)
                 + ADAPTIVE_W_RRF * _minmax(rrf_scores))

        # the cross-lingual consensus: a lower bound for the gold count
        n_agree = sum(1 for p in positions
                      if len(lang_membership.get(p, ())) >= ADAPTIVE_DOC_AGREE)

        if ADAPTIVE_DOCS:
            # the estimator works on the DESCENDING curve -> the curve is sorted
            # for the estimate only; the documents keep the proven RRF order.
            curve_sorted = np.sort(curve)[::-1]
            n_keep, diag = estimate_count(
                curve_sorted,
                rel_cut=ADAPTIVE_DOC_REL_CUT,
                break_w=ADAPTIVE_DOC_BREAK_W,
                shrink=ADAPTIVE_DOC_SHRINK,
                floor=n_agree,
                lo=ADAPTIVE_DOC_MIN,
                hi=ADAPTIVE_DOC_MAX,
                target=ADAPTIVE_DOC_TARGET)
            # ------------------------------------------------------------------ #
            # THE FLOOR RULE (added after the v2 measurement)
            # ------------------------------------------------------------------ #
            # The v2 run LOST 0.030 on the document half because the unsupervised
            # estimator cut the tail of the curve.  The measurement showed why:
            #
            #   * the 22 documents that were dropped were worth 4.46 hits
            #     -> their marginal hit rate was 19.5 %
            #   * the break-even was only 7.2 %
            #   -> dropping them was 2.7x worse than keeping them
            #
            # The reason: the relative score curve is FLAT (0.33 at rank 75), so
            # its "break" is NOISE, not the end of the relevant set.  The
            # estimator cannot tell where the gold ends, so it must not be
            # allowed to CUT the tail.
            #
            # The estimator is therefore only allowed to make the count BIGGER
            # than the floor.  The floor is the count that was measured to work.
            if n_keep < ADAPTIVE_DOC_MIN:
                diag['floored'] = True
                n_keep = ADAPTIVE_DOC_MIN
            else:
                diag['floored'] = False
        else:
            n_keep = min(len(positions), FINAL_TOPK_DOCS)
            diag = {'why': 'fixed'}
        n_keep = int(max(1, min(n_keep, len(positions))))

        doc_ids_out = [self.corpus.ids[p] for p in positions[:n_keep]]
        # the scores are in the SAME order as the documents (the RRF order)
        scores_out = curve[:n_keep].tolist()
        diag['n_agree'] = n_agree
        diag['n_docs'] = n_keep
        return doc_ids_out, scores_out, diag
