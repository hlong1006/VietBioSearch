"""Build or restore the BM25 indexes per language."""

import gc
import hashlib
import shutil

import bm25s
from tqdm.auto import tqdm

from .text import MEDICAL_STOPWORDS, TOKEN_RE, tokenize


# IMPORTANT: the documents and the queries MUST be tokenized the same way.
# bm25s.tokenize() keeps the accents, while our tokenizer folds them -> the
# Vietnamese index would never match a Vietnamese query.  So the documents are
# tokenized HERE with our tokenizer and the token lists are handed to bm25s.

TOKENIZER_SIGNATURE = hashlib.md5(
    ('|'.join([TOKEN_RE.pattern, ','.join(sorted(MEDICAL_STOPWORDS))])).encode('utf-8')
).hexdigest()[:12]


class _TokenStream:
    """Re-iterable lazy token stream for bm25s.

    bm25s makes two passes over a token corpus: one to build the
    vocabulary and one to convert tokens to integer ids.  A normal
    generator would be exhausted after the first pass.  This small
    re-iterable object tokenizes one document at a time and therefore
    avoids keeping hundreds of thousands of Python token lists in RAM.
    """
    def __init__(self, index_list, doc_contents):
        self.index_list = index_list
        self.doc_contents = doc_contents

    def __iter__(self):
        for position in tqdm(self.index_list, desc='tokenize', leave=False):
            tokens = tokenize(self.doc_contents[position], keep_short=True)
            yield tokens if tokens else ['_empty_']

    def __len__(self):
        return len(self.index_list)


def _build_bm25(index_list, doc_contents):
    """Index positions lazily to keep peak RAM bounded."""
    if not index_list:
        return None
    retriever = bm25s.BM25()
    # Do not build `tokenized = [...]`: for Vietnamese this retained
    # millions of Python strings and could exhaust Kaggle RAM.
    retriever.index(_TokenStream(index_list, doc_contents), show_progress=True)
    gc.collect()
    return retriever


def _bm25_is_complete(d):
    """A BM25 folder is only usable if every required file is there AND it was
    built with the SAME tokenizer (a stale index would never match)."""
    if not ((d / 'data.csc.index.npy').exists()
            and (d / 'params.index.json').exists()
            and (d / 'vocab.index.json').exists()):
        return False
    signature = d / 'tokenizer.sig'
    if not signature.exists():
        # An index produced by an older run (before the signature file
        # existed).  Rebuilding the whole index costs ~15 min, so the index is
        # accepted and the signature is written again right after the load.
        return True
    try:
        return signature.read_text(encoding='utf-8').strip() == TOKENIZER_SIGNATURE
    except OSError:
        return False


def load_bm25_indexes(corpus, checkpoint):
    BM25_DIR = checkpoint.work_dir / 'bm25'
    BM25_DIR.mkdir(parents=True, exist_ok=True)

    bm25_indexes = {}

    print('tokenizer signature:', TOKENIZER_SIGNATURE)

    for lang in ('vi', 'en', 'zh'):
        idx = corpus.language_indices.get(lang, [])
        if not idx:
            bm25_indexes[lang] = None
            continue

        save_dir = BM25_DIR / lang

        # 1) reuse the index of a previous session if it was uploaded as a dataset
        restored = False
        if not _bm25_is_complete(save_dir):
            for base in checkpoint.input_dirs:
                cand = base / 'bm25' / lang
                if _bm25_is_complete(cand):
                    try:
                        shutil.copytree(cand, save_dir, dirs_exist_ok=True)
                        restored = True
                    except Exception as exc:
                        print('  could not copy the BM25 index of {}: {}'.format(lang, exc))
                    break

        if restored or _bm25_is_complete(save_dir):
            try:
                bm25_indexes[lang] = bm25s.BM25.load(str(save_dir), mmap=True)
                if not (save_dir / 'tokenizer.sig').exists():
                    (save_dir / 'tokenizer.sig').write_text(
                        TOKENIZER_SIGNATURE, encoding='utf-8')
                print('  reloaded BM25 for', lang)
                continue
            except Exception as exc:
                print('  could not load the BM25 index of {}: {}'.format(lang, exc))

        # 2) build it from scratch
        print('building BM25 for {} ({:,} docs) ...'.format(lang, len(idx)))
        bm25_indexes[lang] = _build_bm25(idx, corpus.contents)
        try:
            bm25_indexes[lang].save(str(save_dir))
            (save_dir / 'tokenizer.sig').write_text(TOKENIZER_SIGNATURE, encoding='utf-8')
        except Exception as exc:
            print('  (BM25 index not saved:', exc, ')')

    checkpoint.set_stage('bm25', status='DONE',
              languages='{} index built'.format(len([l for l in bm25_indexes if bm25_indexes[l]])),
              files='bm25/')
    return bm25_indexes
