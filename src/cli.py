"""Command-line entry point for local and Kaggle runs."""

import argparse
from pathlib import Path

from .config import LIMIT_QUERIES, MODEL_NAME


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return number


def _discover(root, pattern, *, directory=False):
    matches = sorted(
        path for path in root.rglob(pattern)
        if (path.is_dir() if directory else path.is_file())
    )
    return matches[0] if matches else None


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run the VietBioSearch retrieval pipeline.')
    parser.add_argument('--input-root', type=Path, help='Root to search for corpus, queries and state')
    parser.add_argument('--corpus-dir', type=Path, help='Directory containing the corpus Parquet shards')
    parser.add_argument('--queries-file', type=Path, help='Expanded multilingual queries in JSON')
    parser.add_argument('--query-csv', type=Path, help='Vietnamese query CSV fallback')
    parser.add_argument('--state-root', type=Path, help='Directory containing previous checkpoint files or archives')
    parser.add_argument('--work-dir', type=Path, help='Writable directory for checkpoints and intermediate results')
    parser.add_argument('--artifact-dir', type=Path, help='Destination directory for the ZIP files')
    parser.add_argument('--model', default=MODEL_NAME, help='BGE-M3 model ID or local model directory')
    parser.add_argument('--limit-queries', type=_positive_int, default=LIMIT_QUERIES,
                        help='Process only the first N queries in a separate default work directory')
    parser.add_argument('--skip-state-export', action='store_true', help='Skip writing pa5_state.zip')
    args = parser.parse_args(argv)

    kaggle_input = Path('/kaggle/input')
    on_kaggle = kaggle_input.is_dir()
    input_root = args.input_root or (kaggle_input if on_kaggle else None)
    corpus_dir = args.corpus_dir
    queries_file = args.queries_file
    query_csv = args.query_csv
    if input_root is not None:
        corpus_dir = corpus_dir or _discover(input_root, 'corpus_shards', directory=True)
        if queries_file is None and query_csv is None:
            queries_file = _discover(input_root, 'queries_expanded.json')
            query_csv = _discover(input_root, 'query.csv')
    else:
        corpus_dir = corpus_dir or Path('Output/corpus_shards')
        if queries_file is None and query_csv is None:
            local_queries = Path('Output/queries_expanded.json')
            queries_file = local_queries if local_queries.is_file() else None
            query_csv = next((path for path in (Path('Data_crawl/query.csv'), Path('query.csv'))
                              if path.is_file()), None)

    if corpus_dir is None or not corpus_dir.is_dir():
        parser.error('corpus_shards directory not found. Set --corpus-dir or --input-root.')
    for path in (queries_file, query_csv):
        if path is not None and not path.is_file():
            parser.error(f'query file not found: {path}')
    if queries_file is None and query_csv is None:
        parser.error('query input not found. Set --queries-file or --query-csv.')

    base = Path('/kaggle/working') if on_kaggle else Path('Output')
    work_name = 'pa5_output' if args.limit_queries is None else f'pa5_output_limit_{args.limit_queries}'
    work_dir = (args.work_dir or base / work_name).resolve()
    default_artifact_dir = base if on_kaggle and not args.work_dir and args.limit_queries is None else work_dir
    artifact_dir = (args.artifact_dir or default_artifact_dir).resolve()
    state_root = args.state_root or input_root

    from .pipeline import run_pipeline

    run_pipeline(
        corpus_dir=corpus_dir,
        queries_file=queries_file,
        query_csv=query_csv,
        work_dir=work_dir,
        artifact_dir=artifact_dir,
        state_root=state_root.resolve() if state_root is not None else None,
        limit_queries=args.limit_queries,
        model_name=args.model,
        export_checkpoint=not args.skip_state_export,
    )
