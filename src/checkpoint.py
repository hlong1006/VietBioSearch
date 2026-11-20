"""Find and persist resumable pipeline state."""

import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

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
    BM25_WEIGHT,
    CHUNK_ADAPTIVE,
    CHUNK_BREAK_W,
    CHUNK_CHAR_BUDGET,
    CHUNK_MAX,
    CHUNK_MAX_LENGTH,
    CHUNK_MIN,
    CHUNK_NESTED_DROP,
    CHUNK_REL_CUT,
    CHUNK_SCALES,
    CHUNK_SHRINK,
    CHUNK_TARGET,
    CHUNK_WHOLE_DOC,
    CHUNK_TOPK,
    CHUNK_WINDOW,
    CHUNK_WINDOW_DECAY,
    CHUNK_WINDOW_MIN,
    CHUNK_WINDOW_TAIL,
    CHUNK_WINDOW_TOP,
    CURVE_BREAK_MIN_RATIO,
    DENSE_PER_DOC_CHUNKS,
    DENSE_PREFILTER,
    DENSE_PREFILTER_DOCS,
    DENSE_PREFILTER_MAX,
    DENSE_TOPK,
    DENSE_WEIGHT,
    DOC_COVERAGE_TOP,
    DOC_PRIOR_WEIGHT,
    FINAL_TOPK_CHUNKS,
    FINAL_TOPK_DOCS,
    EMBED_MAX_LENGTH,
    EMBED_SUBSAMPLE,
    LANG_WEIGHTS,
    MAX_CHUNKS_PER_DOC,
    MODEL_NAME,
    RERANK_TOPK,
    RRF_K,
)


STATE_FILE_NAME = 'checkpoint.txt'
CKPT_STAGES = ('embed', 'bm25', 'retrieval', 'chunk', 'submission')

MAX_STATE_DEPTH = 8


def _unpack_state(input_root, extract_dir, work_dir):
    """Find the state of a previous session inside /kaggle/input.

    The Kaggle input tree is NOT flat:

        /kaggle/input/datasets/<owner>/<dataset>/...

    so the state can sit at any depth.  Every directory that holds a
    `checkpoint.txt` is a state directory, and a zip is unpacked first if it
    contains the state.

    Layouts that work:
      a) /kaggle/input/<ds>/checkpoint.txt
      b) /kaggle/input/<ds>/pa5_output/checkpoint.txt   (a downloaded Output)
      c) /kaggle/input/<ds>/.../pa5_state.zip           (the zip of cell 15)
      d) any depth in between, e.g.
         /kaggle/input/datasets/<owner>/<ds>/results/pa5_output/checkpoint.txt
    """
    found = []
    if input_root is None or not input_root.exists():
        return found

    # --- walk the WHOLE input tree (the Kaggle tree is not flat) ---------- #
    # the heavy directories are pruned: they hold thousands of files and can
    # never contain the state
    SKIP_DIRS = {'corpus_shards', 'chunk_shards', 'emb_shards', 'bm25',
                 '__MACOSX', '.ipynb_checkpoints'}

    state_dirs = []
    zip_files = []
    root_depth = len(Path(input_root).parts)

    for dirpath, dirnames, filenames in os.walk(input_root):
        d = Path(dirpath)
        if d == work_dir or d == extract_dir:
            dirnames[:] = []
            continue
        depth = len(d.parts) - root_depth
        if depth >= MAX_STATE_DEPTH:
            dirnames[:] = []
            continue
        # prune the heavy branches, keep the rest sorted
        dirnames[:] = sorted(x for x in dirnames if x not in SKIP_DIRS)
        if (d / STATE_FILE_NAME).exists():
            state_dirs.append(d)
            dirnames[:] = []          # the state is found, no need to go deeper
        for f in filenames:
            if f.endswith('.zip'):
                zip_files.append(d / f)

    found.extend(state_dirs)

    # --- a zip that has to be unpacked first ------------------------------- #
    for z in sorted(zip_files):
        # only unpack a zip that really contains our state, otherwise we
        # would waste time unpacking unrelated archives
        try:
            with zipfile.ZipFile(z) as zf:
                names = zf.namelist()
        except Exception as exc:
            print('  could not open', z.name, exc)
            continue
        if not any(n.endswith(STATE_FILE_NAME) for n in names):
            continue

        target = extract_dir / z.relative_to(input_root).with_suffix('')
        if not target.exists():
            print('unpacking', z.name, '->', target)
            try:
                with zipfile.ZipFile(z) as zf:
                    pending = target.with_name(target.name + '.partial')
                    zf.extractall(pending)
                    pending.replace(target)
            except Exception as exc:
                print('  could not unpack:', exc)
                continue

        # the zip may hold the state at the top level or one level deep
        if (target / STATE_FILE_NAME).exists():
            found.append(target)
            continue
        inner = target / 'pa5_state'
        if inner.is_dir():
            found.append(inner)
            continue
        for sub in sorted(target.iterdir()):
            if (sub / STATE_FILE_NAME).exists():
                found.append(sub)
                break
        else:
            for dirpath, dirnames, filenames in os.walk(target):
                if STATE_FILE_NAME in filenames:
                    found.append(Path(dirpath))
                    break

    # de-duplicate, keep the order
    seen, uniq = set(), []
    for d in found:
        key = str(d)
        if key not in seen:
            seen.add(key)
            uniq.append(d)
    return uniq


def load_jsonl(path):
    """Read a JSONL file, silently skipping lines truncated by a kill."""
    rows, bad = [], 0
    if path is None or not Path(path).exists():
        return rows
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print('  skipped {} truncated line(s) in {}'.format(bad, Path(path).name))
    return rows


def append_jsonl(path, rows):
    """Append rows to a JSONL file and flush to disk (crash safe).

    IMPORTANT: if the file was truncated by a kill, its last line has no
    newline. Appending right after it would glue the new JSON onto the broken
    line and the new row would be lost too. So we first check whether the file
    ends with a newline and add one if it does not.
    """
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if path.exists() and path.stat().st_size > 0:
        with open(path, 'rb') as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b'\n'
    with open(path, 'a', encoding='utf-8') as f:
        if needs_newline:
            f.write('\n')
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def _fmt(v):
    if isinstance(v, (list, tuple, dict)):
        return '{} item(s)'.format(len(v))
    return str(v)


def _manifest_config():
    return {
        'BM25_TOPK': BM25_TOPK, 'DENSE_TOPK': DENSE_TOPK,
        'RERANK_TOPK': RERANK_TOPK, 'FINAL_TOPK_DOCS': FINAL_TOPK_DOCS,
        'LANG_WEIGHTS': LANG_WEIGHTS, 'RRF_K': RRF_K,
        'ADAPTIVE_DOCS': ADAPTIVE_DOCS, 'ADAPTIVE_DOC_MIN': ADAPTIVE_DOC_MIN,
        'ADAPTIVE_DOC_MAX': ADAPTIVE_DOC_MAX, 'ADAPTIVE_DOC_TARGET': ADAPTIVE_DOC_TARGET,
        'ADAPTIVE_DOC_REL_CUT': ADAPTIVE_DOC_REL_CUT,
        'CHUNK_SCALES': str(CHUNK_SCALES), 'MAX_CHUNKS_PER_DOC': MAX_CHUNKS_PER_DOC,
        'DOC_COVERAGE_TOP': DOC_COVERAGE_TOP,
        'CHUNK_ADAPTIVE': CHUNK_ADAPTIVE, 'CHUNK_MIN': CHUNK_MIN,
        'CHUNK_MAX': CHUNK_MAX, 'CHUNK_TARGET': CHUNK_TARGET,
        'FINAL_TOPK_CHUNKS': FINAL_TOPK_CHUNKS, 'BM25_WEIGHT': BM25_WEIGHT,
        'DENSE_WEIGHT': DENSE_WEIGHT, 'DOC_PRIOR_WEIGHT': DOC_PRIOR_WEIGHT,
        'DENSE_PREFILTER': DENSE_PREFILTER, 'MODEL_NAME': MODEL_NAME,
    }


RETRIEVAL_CFG = {
    'BM25_TOPK': BM25_TOPK, 'DENSE_TOPK': DENSE_TOPK,
    'RERANK_TOPK': RERANK_TOPK, 'LANG_WEIGHTS': LANG_WEIGHTS, 'RRF_K': RRF_K,
    'ADAPTIVE_DOCS': ADAPTIVE_DOCS, 'ADAPTIVE_DOC_MIN': ADAPTIVE_DOC_MIN,
    'ADAPTIVE_DOC_MAX': ADAPTIVE_DOC_MAX, 'ADAPTIVE_DOC_TARGET': ADAPTIVE_DOC_TARGET,
    'ADAPTIVE_DOC_REL_CUT': ADAPTIVE_DOC_REL_CUT,
    'ADAPTIVE_DOC_BREAK_W': ADAPTIVE_DOC_BREAK_W,
    'ADAPTIVE_DOC_SHRINK': ADAPTIVE_DOC_SHRINK,
    'ADAPTIVE_DOC_AGREE': ADAPTIVE_DOC_AGREE, 'ADAPTIVE_POOL': ADAPTIVE_POOL,
    'ADAPTIVE_W_DENSE': ADAPTIVE_W_DENSE, 'ADAPTIVE_W_BM25': ADAPTIVE_W_BM25,
    'ADAPTIVE_W_RRF': ADAPTIVE_W_RRF, 'MODEL_NAME': MODEL_NAME,
    'CURVE_BREAK_MIN_RATIO': CURVE_BREAK_MIN_RATIO,
}
CHUNK_CFG = {
    'CHUNK_SCALES': str(CHUNK_SCALES), 'MAX_CHUNKS_PER_DOC': MAX_CHUNKS_PER_DOC,
    'BM25_WEIGHT': BM25_WEIGHT, 'DENSE_WEIGHT': DENSE_WEIGHT,
    'DOC_PRIOR_WEIGHT': DOC_PRIOR_WEIGHT, 'DENSE_PREFILTER': DENSE_PREFILTER,
    'DENSE_PREFILTER_DOCS': DENSE_PREFILTER_DOCS,
    'DENSE_PER_DOC_CHUNKS': DENSE_PER_DOC_CHUNKS,
    'DENSE_PREFILTER_MAX': DENSE_PREFILTER_MAX,
    'CHUNK_ADAPTIVE': CHUNK_ADAPTIVE, 'CHUNK_MIN': CHUNK_MIN,
    'CHUNK_MAX': CHUNK_MAX, 'CHUNK_TARGET': CHUNK_TARGET,
    'CHUNK_REL_CUT': CHUNK_REL_CUT, 'CHUNK_BREAK_W': CHUNK_BREAK_W,
    'CHUNK_SHRINK': CHUNK_SHRINK, 'CHUNK_MAX_LENGTH': CHUNK_MAX_LENGTH,
    'CHUNK_NESTED_DROP': CHUNK_NESTED_DROP,
    'CHUNK_WHOLE_DOC': CHUNK_WHOLE_DOC,
    'CHUNK_WINDOW': CHUNK_WINDOW, 'CHUNK_WINDOW_TOP': CHUNK_WINDOW_TOP,
    'CHUNK_WINDOW_TAIL': CHUNK_WINDOW_TAIL,
    'CHUNK_WINDOW_DECAY': CHUNK_WINDOW_DECAY,
    'CHUNK_WINDOW_MIN': CHUNK_WINDOW_MIN,
    'CHUNK_CHAR_BUDGET': CHUNK_CHAR_BUDGET,
    'CURVE_BREAK_MIN_RATIO': CURVE_BREAK_MIN_RATIO,
    'CHUNK_TOPK': CHUNK_TOPK, 'DOC_COVERAGE_TOP': DOC_COVERAGE_TOP,
    'FINAL_TOPK_CHUNKS': FINAL_TOPK_CHUNKS,
    'MODEL_NAME': MODEL_NAME,
}


def _hash_of(d):
    return hashlib.md5(json.dumps(d, sort_keys=True,
                                  ensure_ascii=False).encode('utf-8')).hexdigest()[:10]


RETRIEVAL_HASH = _hash_of(RETRIEVAL_CFG)
CHUNK_HASH = _hash_of(CHUNK_CFG)


def input_signature(corpus_dir, queries_file, query_csv, model_name):
    """Identify the bytes and order of the inputs that produce cached state."""
    shards = sorted(corpus_dir.glob('*.parquet'))
    model_dir = Path(model_name)
    files = {
        'shards': shards,
        'queries': [path for path in (queries_file, query_csv) if path is not None],
        'model': sorted(path for path in model_dir.iterdir() if path.is_file())
        if model_dir.is_dir() else [],
    }
    signature = {}
    for category, paths in files.items():
        entries = []
        for path in paths:
            with path.open('rb') as handle:
                entries.append((path.name, hashlib.file_digest(handle, 'sha256').hexdigest()))
        signature[category] = entries
    signature['model_id'] = str(model_name) if not model_dir.is_dir() else None
    signature['embedding'] = [EMBED_MAX_LENGTH, EMBED_SUBSAMPLE]
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode('utf-8')).hexdigest()


class Checkpoint:
    def __init__(self, work_dir, state_root, extract_dir, signature):
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        local_manifest = self.work_dir / STATE_FILE_NAME
        local = self.read_manifest(local_manifest)
        if local and local['meta'].get('input_signature') != signature:
            raise ValueError(f'Incompatible checkpoint in {work_dir}. Choose a new --work-dir.')
        if local is None and any((work_dir / name).exists() for name in (
                'bm25', 'emb_shards', 'doc_emb.npy', 'retrieval.jsonl',
                'retrieval_results.json', 'chunks.jsonl', 'chunk_checkpoint.json')):
            raise ValueError(f'Unverified state in {work_dir}. Choose a new --work-dir.')
        self.input_dirs = []
        for base in _unpack_state(state_root, extract_dir, work_dir):
            source = self.read_manifest(base / STATE_FILE_NAME)
            if source and source['meta'].get('input_signature') == signature:
                self.input_dirs.append(base)
            else:
                print('skipping checkpoint with incompatible inputs:', base)
        print('state datasets found:', [d.name for d in self.input_dirs] or 'NONE')
        self.previous = self.read_manifest()
        self.data = {'meta': {}, 'config': _manifest_config(), 'stages': {}}
        self.data['meta'].update(
            notebook='pa5_best_kaggle_v2',
            config_hash=self.config_hash(),
            retrieval_hash=RETRIEVAL_HASH,
            chunk_hash=CHUNK_HASH,
            input_signature=signature,
        )
        if self.previous:
            self.data['meta']['resumed_from'] = self.previous['_path']
            print('resuming from:', self.previous['_path'])

        # Clear incompatible results before publishing the new manifest.
        previous_meta = self.previous['meta'] if self.previous else {}
        if previous_meta.get('retrieval_hash') not in (None, RETRIEVAL_HASH):
            self._remove_results('retrieval.jsonl', 'retrieval_results.json')
        if (previous_meta.get('chunk_hash') != CHUNK_HASH
                or previous_meta.get('retrieval_hash') != RETRIEVAL_HASH):
            self._remove_results('chunks.jsonl', 'chunk_checkpoint.json', 'submission.json')
        self.save_manifest()
        print('checkpoint file:', self.work_dir / STATE_FILE_NAME)
        print('retrieval hash:', RETRIEVAL_HASH, '| chunk hash:', CHUNK_HASH)

    def _remove_results(self, *names):
        for name in names:
            path = self.work_dir / name
            if path.exists():
                path.unlink()
                print('removed stale result:', name)

    def result_path(self, name, *, chunks=False):
        """Locate results from a compatible checkpoint."""
        for base in [self.work_dir] + self.input_dirs:
            path = base / name
            if not path.is_file() or not path.stat().st_size:
                continue
            if base != self.work_dir:
                source = self.read_manifest(base / STATE_FILE_NAME)
                meta = source['meta'] if source else {}
                if meta.get('retrieval_hash') not in (None, RETRIEVAL_HASH):
                    continue
                if chunks and (meta.get('retrieval_hash') != RETRIEVAL_HASH
                               or meta.get('chunk_hash') != CHUNK_HASH):
                    continue
            return path
        return None

    def materialize_result(self, name, *, chunks=False):
        destination = self.work_dir / name
        source = self.result_path(name, chunks=chunks)
        if source is not None and source != destination:
            temporary = destination.with_suffix('.tmp')
            shutil.copy2(source, temporary)
            temporary.replace(destination)
            print('restored:', source)
        return destination

    def resolve_state(self, name):
        """Existing state file: this session first, then the uploaded dataset."""
        for base in [self.work_dir] + self.input_dirs:
            p = base / name
            if p.exists() and p.stat().st_size > 0:
                return p
        return None


    def resolve_with_marker(self, name, marker):
        """Same as resolve_state but only if the marker file sits next to it.

        A marker proves the file was written completely. Without it a half written
        file would be reloaded as if it were fine and would silently corrupt the
        results.
        """
        for base in [self.work_dir] + self.input_dirs:
            p, m = base / name, base / marker
            if p.exists() and m.exists() and p.stat().st_size > 0:
                return p
        return None


    def config_hash(self):
        blob = json.dumps(self.data['config'], sort_keys=True, ensure_ascii=False)
        return hashlib.md5(blob.encode('utf-8')).hexdigest()[:10]


    def save_manifest(self):
        """Write the human readable checkpoint.txt."""
        lines = []
        lines.append('# ============================================================')
        lines.append('# PA5 CHECKPOINT - state of the pipeline')
        lines.append('# updated: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
        lines.append('# Read this file to know how far the previous session got.')
        lines.append('# The actual data lives in the files listed in each stage.')
        lines.append('# ============================================================')
        lines.append('')
        lines.append('[META]')
        for k in ('notebook', 'config_hash', 'input_signature', 'total_queries', 'corpus_docs',
                  'resumed_from', 'retrieval_hash', 'chunk_hash'):
            if k in self.data['meta']:
                lines.append('{} = {}'.format(k, _fmt(self.data['meta'][k])))
        lines.append('')
        lines.append('[CONFIG]')
        for k, v in self.data['config'].items():
            lines.append('{} = {}'.format(k, _fmt(v)))
        lines.append('')
        for stage in CKPT_STAGES:
            lines.append('[STAGE:{}]'.format(stage))
            fields = self.data['stages'].get(stage)
            if not fields:
                lines.append('status = PENDING')
            else:
                for k, v in fields.items():
                    lines.append('{} = {}'.format(k, _fmt(v)))
            lines.append('')
        text = '\n'.join(lines)
        path = self.work_dir / STATE_FILE_NAME
        temporary = path.with_suffix('.tmp')
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(path)
        return text


    def set_stage(self, stage, **fields):
        fields.setdefault('updated', time.strftime('%Y-%m-%d %H:%M:%S'))
        self.data['stages'][stage] = fields
        self.save_manifest()
        return fields


    def read_manifest(self, path=None):
        """Parse a checkpoint.txt back into a dict."""
        p = Path(path) if path is not None else self.resolve_state(STATE_FILE_NAME)
        if p is None or not p.exists():
            return None
        out = {'meta': {}, 'config': {}, 'stages': {}}
        section = None
        for raw in p.read_text(encoding='utf-8').splitlines():
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('[') and line.endswith(']'):
                name = line[1:-1]
                if name.startswith('STAGE:'):
                    section = out['stages'].setdefault(name[6:], {})
                elif name in ('META', 'CONFIG'):
                    section = out[name.lower()]
                else:
                    section = None
                continue
            if section is not None and '=' in line:
                k, v = line.split('=', 1)
                section[k.strip()] = v.strip()
        out['_path'] = str(p)
        return out
