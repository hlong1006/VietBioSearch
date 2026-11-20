"""Multilingual tokenization and multi-scale document chunking."""

import re

from .config import (
    CHUNK_NATURAL_CHARS,
    CHUNK_SCALES,
)


TOKEN_RE = re.compile(r'[\u4e00-\u9fff]|[^\W_]+', re.UNICODE)

MEDICAL_STOPWORDS = set("""
cua va co trong voi cho de la cac nhung mot nhieu ban toi khong duoc
the of and to in a for is on with are was were be been this that it its
zhi shi le you wei he yu zhong bei de y cao moi
""".split())


def _units(text):
    """(token, start, end) for every token of the text."""
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text or '')]


def tokenize(text, keep_short=False):
    if not text:
        return []
    toks = [t.lower() for t in TOKEN_RE.findall(text)]
    if keep_short:
        return [t for t in toks if t not in MEDICAL_STOPWORDS]
    return [t for t in toks if len(t) > 1 and t not in MEDICAL_STOPWORDS]


def chunk_fixed(text, max_tokens, overlap_tokens):
    """Fixed-size token chunks with a token overlap.

    Every returned chunk is a verbatim substring of `text`.
    """
    units = _units(text)
    n = len(units)
    if n == 0:
        return []
    if n <= max_tokens:
        return [(text.strip(), tokenize(text, keep_short=True))]

    step = max(1, max_tokens - overlap_tokens)
    tail_ratio = 0.35
    out = []
    i = 0
    while i < n:
        j = min(i + max_tokens, n)
        if j < n and (n - j) < max_tokens * tail_ratio:
            j = n                                  # absorb the short tail
        chunk_text = text[units[i][1]:units[j - 1][2]].strip()
        if chunk_text:
            out.append((chunk_text, tokenize(chunk_text, keep_short=True)))
        if j >= n:
            break
        i += step
    return out


_SENT_RE = re.compile(r'(?<=[.!?。！？])\s*|\n{1,}')


def chunk_natural(text, max_chars):
    """Sentence-boundary chunks of at most `max_chars` characters.

    Every returned chunk is a verbatim substring of `text`.
    """
    text = (text or '').strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [(text, tokenize(text, keep_short=True))]

    # Split while keeping the offsets so the chunks stay verbatim.
    spans = []
    start = 0
    for m in _SENT_RE.finditer(text):
        end = m.end()
        if end - start >= 1:
            spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))

    # hard-split the spans that are longer than max_chars (a single very long
    # sentence must not produce an oversized chunk)
    hard = []
    for s, e in spans:
        pos = s
        while e - pos > max_chars:
            hard.append((pos, pos + max_chars))
            pos += max_chars
        if e > pos:
            hard.append((pos, e))
    spans = hard

    out = []
    cur_start = None
    cur_end = None
    for s, e in spans:
        if cur_start is None:
            cur_start, cur_end = s, e
            continue
        if e - cur_start <= max_chars:
            cur_end = e
            continue
        out.append((cur_start, cur_end))
        cur_start, cur_end = s, e
    if cur_start is not None:
        out.append((cur_start, cur_end))

    chunks = []
    for s, e in out:
        piece = text[s:e].strip()
        if piece:
            chunks.append((piece, tokenize(piece, keep_short=True)))
    return chunks


def chunk_document_multiscale(text):
    """All scales of one document -> list of dicts with a verbatim text."""
    out = []
    for scale in CHUNK_SCALES:
        if scale[0] == 'natural':
            produced = chunk_natural(text, CHUNK_NATURAL_CHARS)
        else:
            produced = chunk_fixed(text, scale[0], scale[1])
        for ctext, ctoks in produced:
            if ctext:
                out.append({'text': ctext, 'tokens': ctoks, 'scale': scale[0]})
    return out
