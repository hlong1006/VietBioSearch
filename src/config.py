"""PA5 v2 ranking, chunking and embedding parameters."""


# Hybrid retrieval
BM25_TOPK        = 300     # BM25 candidates per language
DENSE_TOPK       = 300     # dense candidates per language
RERANK_TOPK      = 100     # documents kept after the dense rerank, per language
FINAL_TOPK_DOCS  = 120     # fallback when ADAPTIVE_DOCS is off
LANG_WEIGHTS     = {'vi': 0.70, 'en': 0.15, 'zh': 0.15}
RRF_K            = 60

# Adaptive document count
ADAPTIVE_DOCS        = True     # False -> always FINAL_TOPK_DOCS
ADAPTIVE_DOC_MIN     = 120      # never fewer than this
ADAPTIVE_DOC_MAX     = 200      # never more than this
ADAPTIVE_DOC_TARGET  = 150      # the value the estimate is shrunk towards
ADAPTIVE_DOC_REL_CUT = 0.55     # relative score below this = not relevant
ADAPTIVE_DOC_BREAK_W = 0.5      # weight of the break vs the threshold
ADAPTIVE_DOC_SHRINK  = 0.30     # how much of the raw estimate is kept
ADAPTIVE_DOC_AGREE   = 2        # a document found by >= 2 languages is confirmed
ADAPTIVE_POOL        = 200      # the candidate pool the estimate works on
ADAPTIVE_W_DENSE     = 0.5      # the weights of the curve that is analysed
ADAPTIVE_W_BM25      = 0.3
ADAPTIVE_W_RRF       = 0.2
CURVE_BREAK_MIN_RATIO = 2.5

# Document embeddings and dense search
EMBED_BATCH_TOKENS = 120000   # dynamic batching budget (tokens per batch)
EMBED_MAX_SEQS     = 64        # max sequences per batch (activation peak -> OOM risk)
DENSE_CHUNK = 32768  # Rows per block when scoring dense embeddings.
EMBED_MAX_LENGTH   = 512
EMBED_SUBSAMPLE    = None     # e.g. 200000 to embed only a part of the corpus

# Chunk generation
CHUNK_FIXED_SCALES  = [(300, 60), (700, 150)]   # (max_tokens, overlap_tokens)
CHUNK_NATURAL_CHARS = 2500                      # sentence-boundary scale
CHUNK_SCALES = CHUNK_FIXED_SCALES + [('natural', CHUNK_NATURAL_CHARS)]
MAX_CHUNKS_PER_DOC = 5
DOC_COVERAGE_TOP   = 10        # top documents guaranteed at least one chunk
CHUNK_NESTED_DROP = True       # drop a chunk literally nested in another one
CHUNK_WHOLE_DOC = True

# Submission windows
CHUNK_WINDOW       = True   # False -> submit the whole documents as before
CHUNK_WINDOW_TOP   = 3600   # window length (chars) of the rank-0 document
CHUNK_WINDOW_TAIL  = 900    # window length (chars) from rank CHUNK_WINDOW_DECAY on
CHUNK_WINDOW_DECAY = 100    # rank at which the length reached the tail value
CHUNK_WINDOW_MIN   = 600    # a window is never shorter than this
CHUNK_CHAR_BUDGET  = 230000 # hard cap of the characters of one query
FINAL_TOPK_CHUNKS        = 80   # fallback when the adaptive budget is off

# Chunk scoring and GPU batching
BM25_WEIGHT              = 0.30
DENSE_WEIGHT             = 0.70
DOC_PRIOR_WEIGHT         = 0.20
DENSE_PREFILTER          = 150   # the BM25-best chunks that are always encoded
CHUNK_MAX_LENGTH         = 1024
CHUNK_ENCODE_BATCH = 12  # Texts per chunk encoding call.
CHUNK_MIN_FREE_VRAM     = 4.0    # GB

# Adaptive chunk count
CHUNK_ADAPTIVE   = True
CHUNK_MIN       = 100      # never fewer than this (the measured value)
CHUNK_MAX       = 260      # never more than this
CHUNK_TARGET    = 150      # the value the estimate is shrunk towards
CHUNK_REL_CUT   = 0.55     # relative score below this = not relevant
CHUNK_BREAK_W   = 0.5      # weight of the break vs the threshold
CHUNK_SHRINK    = 0.30     # how much of the raw estimate is kept
CHUNK_TOPK      = None     # None -> the adaptive budget is used

# Dense pre-filter coverage
DENSE_PREFILTER_DOCS = 150     # the best N documents whose chunks are encoded
DENSE_PER_DOC_CHUNKS = 4       # the best N chunks of each of those documents
DENSE_PREFILTER_MAX  = 1200    # the hard cap of the encoded chunks per query

# Run defaults
CHECKPOINT_EVERY = 10
LIMIT_QUERIES    = None
MODEL_NAME       = 'BAAI/bge-m3'
