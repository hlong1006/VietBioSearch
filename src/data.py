"""Load corpus shards and multilingual queries."""

import gc
import json
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


@dataclass
class Corpus:
    ids: list[str]
    contents: list[str]
    positions: dict[str, int]
    language_indices: dict[str, list[int]]
    shard_offsets: np.ndarray


def load_corpus(corpus_dir):
    shards = sorted(corpus_dir.glob('*.parquet'))
    if not shards:
        raise FileNotFoundError(f'no parquet shard inside {corpus_dir}')

    doc_ids, doc_contents, doc_langs = [], [], []
    shard_sizes = []
    for shard in tqdm(shards, desc='corpus shards'):
        df = pd.read_parquet(shard, columns=['id', 'content', 'language'])
        if df['id'].isna().any():
            raise ValueError(f'missing document ID in {shard}')
        shard_sizes.append(len(df))
        doc_ids.extend(df['id'].astype(str).tolist())
        doc_contents.extend(df['content'].fillna('').astype(str).tolist())
        doc_langs.extend(df['language'].fillna('vi').astype(str).tolist())
        del df

    gc.collect()
    n_docs = len(doc_ids)
    doc_pos = {d: i for i, d in enumerate(doc_ids)}
    if not n_docs or len(doc_pos) != n_docs or '' in doc_pos:
        raise ValueError('Corpus must contain documents with unique, non-empty IDs.')
    lang_indices = defaultdict(list)
    for i, lang in enumerate(doc_langs):
        lang_indices[lang].append(i)
    for lang in sorted(lang_indices):
        print(f'  {lang}: {len(lang_indices[lang]):,} docs')
    print(f'{n_docs:,} documents loaded')

    # the row range of every shard inside the global arrays -> needed to store the
    # embeddings shard by shard and to rebuild them later
    shard_offsets = np.zeros(len(shard_sizes) + 1, dtype=np.int64)
    shard_offsets[1:] = np.cumsum(shard_sizes)
    print('shard offsets ready:', len(shards), 'shards')

    return Corpus(doc_ids, doc_contents, doc_pos, dict(lang_indices), shard_offsets)


def load_queries(queries_file, query_csv, limit=None):
    queries = []

    if queries_file is not None and queries_file.exists():
        raw = json.loads(queries_file.read_text(encoding='utf-8'))
        if isinstance(raw, dict):
            if not all(isinstance(q, dict) for q in raw.values()):
                raise ValueError('Each expanded query must be an object.')
            queries = [{'id': qid, **q} for qid, q in raw.items()]
        elif isinstance(raw, list):
            queries = raw
        else:
            raise ValueError('Query JSON must be an object or a list.')
        print('queries loaded from', queries_file.name)

    if not queries and query_csv is not None:
        df = pd.read_csv(query_csv, dtype=str, keep_default_na=False)
        if len(df.columns) < 2:
            raise ValueError('Query CSV must contain ID and query columns.')
        id_col = 'id' if 'id' in df.columns else df.columns[0]
        q_col = 'query' if 'query' in df.columns else df.columns[-1]
        queries = [{'id': str(r[id_col]), 'vi_orig': str(r[q_col]), 'en': '', 'zh': ''}
                   for _, r in df.iterrows()]
        print('queries built from', query_csv.name)

    if not queries:
        raise FileNotFoundError('no queries found (queries_expanded.json / query.csv)')

    seen = set()
    fields = ('vi_orig', 'query', 'vi_exp', 'en', 'zh')
    for position, query in enumerate(queries):
        if not isinstance(query, dict):
            raise ValueError(f'Query at position {position} must be an object.')
        raw_id = query.get('id')
        if isinstance(raw_id, bool) or not isinstance(raw_id, (int, str)):
            raise ValueError(f'Query at position {position} must have an integer ID.')
        try:
            query_id = int(raw_id)
        except ValueError as exc:
            raise ValueError(f'Invalid query ID at position {position}: {raw_id!r}') from exc
        if query_id in seen:
            raise ValueError(f'Duplicate query ID: {query_id}')
        seen.add(query_id)
        query['id'] = query_id
        for field in fields:
            value = query.get(field)
            if value is not None and not isinstance(value, str):
                raise ValueError(f'Query {query_id}: {field} must be text.')
        if not any((query.get(field) or '').strip() for field in fields):
            raise ValueError(f'Query {query_id} has no text.')

    if limit:
        queries = queries[:limit]

    print(f'{len(queries)} queries ready')
    return queries
