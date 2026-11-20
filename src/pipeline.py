"""Orchestrate corpus loading, retrieval, chunk selection and export."""

import os

from .config import MODEL_NAME


def run_pipeline(
    *, corpus_dir, queries_file, query_csv, work_dir, artifact_dir,
    state_root=None, limit_queries=None, export_checkpoint=True, model_name=MODEL_NAME,
):
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    import torch

    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU is required. On Kaggle, select a GPU accelerator.')
    device = 'cuda'
    print('GPU:', torch.cuda.get_device_name(0))

    from FlagEmbedding import BGEM3FlagModel

    from .bm25_index import load_bm25_indexes
    from .checkpoint import Checkpoint, input_signature
    from .chunk_selection import ChunkSelector
    from .data import load_corpus, load_queries
    from .embeddings import load_document_embeddings
    from .export import export_state, export_submission
    from .retrieval import DocumentRetriever
    from .stages import run_chunk_selection, run_retrieval

    print('corpus dir  :', corpus_dir)
    print('queries file:', queries_file)
    print('query csv   :', query_csv)
    queries = load_queries(queries_file, query_csv, limit_queries)

    signature = input_signature(corpus_dir, queries_file, query_csv, model_name)
    checkpoint = Checkpoint(work_dir, state_root, work_dir / '_state_extract', signature)
    corpus = load_corpus(corpus_dir)
    checkpoint.data['meta']['corpus_docs'] = len(corpus.ids)
    checkpoint.save_manifest()

    model = BGEM3FlagModel(model_name, use_fp16=True, devices=device)
    print('model ready:', model_name)
    bm25_indexes = load_bm25_indexes(corpus, checkpoint)
    embeddings = load_document_embeddings(corpus, model, device, checkpoint)
    retriever = DocumentRetriever(corpus, bm25_indexes, embeddings, model, device)
    retrieval_results = run_retrieval(queries, retriever, checkpoint)

    # The chunk stage uses the model and corpus, so release the document indexes and GPU matrices.
    del retriever, embeddings, bm25_indexes
    queries_by_id = {str(query['id']): query for query in queries}
    submission = run_chunk_selection(
        retrieval_results, queries_by_id, ChunkSelector(corpus, model), checkpoint,
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    submission_zip = export_submission(submission, checkpoint, artifact_dir)
    if export_checkpoint:
        matrix = checkpoint.resolve_with_marker('doc_emb.npy', 'doc_emb.done')
        export_state(work_dir, artifact_dir, embedding_matrix=matrix)
    return submission_zip
