"""Choose bounded, source-aligned text windows for selected chunks."""

from .config import (
    CHUNK_CHAR_BUDGET,
    CHUNK_WINDOW_DECAY,
    CHUNK_WINDOW_MIN,
    CHUNK_WINDOW_TAIL,
    CHUNK_WINDOW_TOP,
    DOC_COVERAGE_TOP,
)


_BOUNDARY_CHARS = '\n.!?\u3002\uff01\uff1f'


def _snap_start(text, lo, limit):
    """Smallest sentence start in [lo, limit); `lo` itself when there is none."""
    if lo <= 0:
        return 0
    for p in range(lo, max(lo, limit)):
        if text[p - 1] in _BOUNDARY_CHARS:
            while p < limit and text[p].isspace():
                p += 1
            return p
    return lo


def _snap_end(text, hi, limit):
    """Largest sentence end in (limit, hi]; `hi` itself when there is none."""
    n = len(text)
    if hi >= n:
        return n
    for p in range(hi, min(hi, limit), -1):
        if text[p - 1] in _BOUNDARY_CHARS:
            return p
    return hi


def _window_span(text, span, length):
    """[lo, hi) of at most ~`length` chars around `span`, inside `text`.

    The span (the best sub-chunk) is always kept when it fits.  Both ends are
    snapped to a sentence boundary, but never into the protected span.
    """
    n = len(text)
    s, e = span
    quarter = max(1, length // 5)
    if e - s >= length:
        centre = (s + e) // 2
        lo = max(0, centre - length // 2)
        hi = min(n, lo + length)
        lo = max(0, hi - length)
        prot_lo, prot_hi = lo + quarter, hi - quarter
    else:
        pad = length - (e - s)
        lo = s - pad // 2
        hi = e + (pad - pad // 2)
        if lo < 0:
            hi = min(n, hi - lo)
            lo = 0
        if hi > n:
            lo = max(0, lo - (hi - n))
            hi = n
        prot_lo, prot_hi = min(s, lo + quarter), max(e, hi - quarter)
    lo = _snap_start(text, lo, prot_lo)
    hi = _snap_end(text, hi, prot_hi)
    return lo, hi


def _window_limit(rank, scale):
    t = min(1.0, rank / float(max(1, CHUNK_WINDOW_DECAY)))
    length = CHUNK_WINDOW_TOP + (CHUNK_WINDOW_TAIL - CHUNK_WINDOW_TOP) * t
    return int(max(CHUNK_WINDOW_MIN, length * scale))


def _apply_windows(cands, selected, hybrid):
    """The selected chunks with every long WHOLE document cut to its window.

    `selected` is sorted by the hybrid score (best first).  Chunks that are not
    whole documents are already query-selected sub-chunks and stay untouched.
    The total number of characters is kept under CHUNK_CHAR_BUDGET: first the
    windows shrink together, then the lowest ranked chunks are dropped.
    """
    best_child = {}
    for i, c in enumerate(cands):
        if c.get('scale') == 'whole_doc':
            continue
        d = c['doc_id']
        if d not in best_child or hybrid[i] > hybrid[best_child[d]]:
            best_child[d] = i

    spans = {}
    for i in selected:
        c = cands[i]
        if c.get('scale') != 'whole_doc':
            continue
        text = c['text']
        bi = best_child.get(c['doc_id'])
        span = None
        if bi is not None:
            pos = text.find(cands[bi]['text'])
            if pos >= 0:
                span = (pos, pos + len(cands[bi]['text']))
        spans[i] = span if span is not None else (0, min(len(text), CHUNK_WINDOW_MIN))

    def build(scale):
        rows, seen = [], set()
        for i in selected:
            c = cands[i]
            text = c['text']
            if i in spans:
                length = _window_limit(c['doc_rank'], scale)
                if len(text) > length:
                    lo, hi = _window_span(text, spans[i], length)
                    text = text[lo:hi].strip() or text[:length]
            if text in seen:
                continue
            seen.add(text)
            rows.append({'doc_id': c['doc_id'], 'chunk_text': text})
        return rows

    scale = 1.0
    rows = build(scale)
    total = sum(len(r['chunk_text']) for r in rows)
    for _ in range(6):
        if total <= CHUNK_CHAR_BUDGET or scale <= 0.35:
            break
        scale = max(0.35, scale * CHUNK_CHAR_BUDGET / float(total) * 0.97)
        rows = build(scale)
        total = sum(len(r['chunk_text']) for r in rows)
    keep = max(1, DOC_COVERAGE_TOP)
    while total > CHUNK_CHAR_BUDGET and len(rows) > keep:
        total -= len(rows.pop()['chunk_text'])
    return rows
