# Lab 1 — Architecture: RAG pipeline over jreypo.io

Technical spec for the lab. It serves as the basis for you to write the
implementation by hand. Each section defines **what** needs to be built and
**why**, without dictating the how in code.

---

## 1. Goal

Build a working RAG pipeline over the corpus of the `jreypo.io` blog and, in
three progressive phases, observe how retrieval quality changes as we change the
chunking strategy and the query strategy:

1. **Phase 1**: naïve chunking (fixed size).
2. **Phase 2**: semantic chunking by Markdown sections.
3. **Phase 3**: HyDE (Hypothetical Document Embeddings).

The point is not to reach the prettiest pipeline but to **measure** the
difference between phases with our own golden set, and to understand why each
change moves the metric.

---

## 2. Stack

| Layer | Choice | Notes |
|---|---|---|
| Embeddings | OVH AI Endpoints — `bge-multilingual-gemma2` | 3584 dims, 8192-token context, multilingual (English corpus, but queries may come in ES/EN). €0.01/M tokens. |
| Vector store | ChromaDB (Docker) | Single-node, volume persistence. Python client in HTTP mode. |
| Generation | OVH AI Endpoints — `gpt-oss-120b` | Same OpenAI-compat SDK. Used for the final answer and for HyDE. |
| Client | `openai` SDK with `base_url` pointing to OVH | OVH exposes an OpenAI-compatible API; a single SDK works for embed + chat. |
| Dev runtime | Python 3.14 in a local venv + Chroma in Docker | Fast iteration. Chroma outside Python avoids restarting when the code changes. |
| Bonus runtime | Kubernetes (homelab) | At the end of the lab: Chroma deployment + ingestion Job + app Deployment. |
| Language | Python ≥ 3.13 | Defined in `pyproject.toml`. |

---

## 3. Component view

```
┌──────────────────────────┐
│ Blog repo (Hugo, .md)    │   BLOG_REPO_PATH=/Users/.../jreypo.github.io
│  content/posts/*.md      │
└────────────┬─────────────┘
             │ filesystem read
             ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│  loader.py               │    │  OVH AI Endpoints        │
│  .md → Post              │    │  (OpenAI-compat HTTPS)   │
└────────────┬─────────────┘    │                          │
             │                  │  ┌────────────────────┐  │
             ▼                  │  │ /v1/embeddings     │  │
┌──────────────────────────┐    │  │ bge-multi-gemma2   │  │
│  chunker.py              │    │  └────────┬───────────┘  │
│  Post → list[Chunk]      │    │           │              │
│  (naive | semantic_md)   │    │  ┌────────▼───────────┐  │
└────────────┬─────────────┘    │  │ /v1/chat/completions│ │
             │                  │  │ gpt-oss-120b        │ │
             │ texts            │  └────────────────────┘  │
             ▼                  └────▲─────────▲───────────┘
┌──────────────────────────┐         │         │
│  ovh_client.py           │─────────┘         │
│  embed() / chat()        │                   │
└────────────┬─────────────┘                   │
             │ vectors                         │
             ▼                                 │
┌──────────────────────────┐                   │
│  ingest.py               │                   │
│  upsert chunks+vectors   │                   │
└────────────┬─────────────┘                   │
             │                                 │
             ▼                                 │
┌──────────────────────────┐                   │
│  ChromaDB (Docker)       │                   │
│  collection per phase    │                   │
└────────────▲─────────────┘                   │
             │ query top-k                     │
             │                                 │
┌────────────┴─────────────┐                   │
│  retriever.py            │                   │
│  dense | hyde            │                   │
└────────────▲─────────────┘                   │
             │ query                           │
┌────────────┴─────────────┐                   │
│  pipeline.py             │───────────────────┘
│  query → retrieve → gen  │
└──────────────────────────┘
```

---

## 4. Data flow

**Ingestion** (offline, once per phase):

```
posts dir → load_all_posts() → [Post]
            → chunk_*()       → [Chunk]
            → embed(batch)    → [vector]
            → chroma.upsert(ids, texts, metadatas, embeddings)
```

**Query** (online, per question):

```
question
  ├─ (phase 1/2) → embed(question)        ──┐
  └─ (phase 3)   → hyde_doc = chat(prompt)  │
                  → embed(hyde_doc)         │
                                            ▼
                              chroma.query(vector, top_k)
                                            │
                                            ▼
                              context = join([chunk.text, ...])
                                            │
                                            ▼
                              chat([sys, user(question + context)])
                                            │
                                            ▼
                                        answer
```

---

## 5. Project structure

What is ALREADY done (scaffold):

```
rag-chunking-lab/
├── .env.example          ← variables to fill in
├── .gitignore            ← .env, .venv/, chroma_data/, __pycache__/
├── docker-compose.yml    ← Chroma service with persistent volume
├── Dockerfile            ← Python image for the k8s phase (CMD to be defined)
├── pyproject.toml        ← deps: openai, chromadb, frontmatter, dotenv...
├── src/
│   ├── __init__.py
│   └── config.py         ← loads .env → Settings dataclass
├── eval/
│   └── golden_set.yaml   ← empty format awaiting questions
├── notebooks/            ← empty
└── k8s/                  ← empty (bonus)
```

What YOU are going to create:

```
src/
├── loader.py             ← reads Hugo .md → Post
├── chunker.py            ← Post → list[Chunk] (2 strategies)
├── ovh_client.py         ← OpenAI-compat wrapper (embed + chat)
├── ingest.py             ← orchestrates posts → chunks → embed → Chroma
├── retriever.py          ← query → top-k chunks (dense | hyde)
└── pipeline.py           ← end-to-end: query → answer

scripts/
└── smoke_test.py         ← checks blog + Chroma + OVH

eval/
└── run_eval.py           ← Recall@k and MRR over golden_set.yaml

notebooks/
├── 01_naive_chunking.ipynb
├── 02_semantic_chunking.ipynb
└── 03_hyde.ipynb
```

---

## 6. `src/` modules — suggested contracts

These are contracts, not implementations. Feel free to vary names or signatures
if, at implementation time, you see a better way.

### `loader.py`

Responsibility: read the `.md` files in the blog's `content/posts/` directory and
produce `Post` objects with parsed frontmatter.

Contract:

```python
@dataclass(frozen=True)
class Post:
    slug: str             # extracted from the filename
    title: str            # from frontmatter
    date: date | None     # from frontmatter or from the filename
    tags: tuple[str, ...]
    body: str             # markdown without frontmatter
    source_path: Path

    @property
    def url_path(self) -> str: ...   # /YYYY/MM/DD/slug/

def load_all_posts(posts_dir: Path) -> list[Post]: ...
```

Details to resolve at implementation time:
- Blog filename: `YYYY-MM-DD-slug.md`. It must be parsed.
- Hugo frontmatter in YAML between `---`. Use `python-frontmatter`.
- Filter out `_index.md` and empty files.
- Tags may come as a string or a list in YAML.

### `chunker.py`

Responsibility: split a `Post` into `Chunk`s. Two separate strategies, one per
phase.

Common contract:

```python
@dataclass(frozen=True)
class Chunk:
    chunk_id: str           # deterministic; see §11
    text: str               # chunk content (markdown)
    post_slug: str
    post_title: str
    post_date: str | None   # ISO 8601
    source_url: str         # https://jreypo.io/YYYY/MM/DD/slug/
    chunk_index: int        # order within the post
    strategy: str           # "naive" | "semantic_md"
    section_heading: str | None  # only in semantic_md

def chunk_post_naive(post: Post, *, chunk_chars: int, overlap: int) -> list[Chunk]: ...
def chunk_post_semantic_md(post: Post, *, max_chars: int) -> list[Chunk]: ...
```

### `ovh_client.py`

Responsibility: a thin wrapper over the `openai` SDK with OVH's `base_url`.

Contract:

```python
class OVHClient:
    def __init__(self, settings: Settings) -> None: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
    # internal batching if len(texts) > BATCH_SIZE
    # retries with exponential backoff on 429/5xx

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str: ...
```

Details to resolve at implementation time:
- Is the OVH base URL for `bge-multilingual-gemma2` and `gpt-oss-120b` the same
  or are they different? Check in the panel. If different, use two clients.
- Batching: 32–64 texts per embeddings call. Check OVH's limit.
- Transient errors: `tenacity` or a simple hand-rolled retry.

### `ingest.py`

Responsibility: orchestrate posts → chunks → embeddings → Chroma. Idempotent.

Pseudo:

```python
def ingest(
    posts: list[Post],
    chunker: Callable[[Post], list[Chunk]],
    ovh: OVHClient,
    chroma_collection,                  # chromadb.Collection
    batch_size: int = 64,
) -> None: ...
```

- Generate chunks with `chunker`.
- Call `ovh.embed()` in batches.
- `collection.upsert(ids=..., documents=..., metadatas=..., embeddings=...)`.
- Idempotency: use `Chunk.chunk_id` → reingesting the same post does not duplicate.

### `retriever.py`

Responsibility: given a query, return the top-k chunks. Two strategies.

Contract:

```python
@dataclass
class Retrieved:
    chunk_id: str
    text: str
    metadata: dict
    score: float           # similarity / 1 - distance

class DenseRetriever:
    def __init__(self, ovh, collection): ...
    def search(self, query: str, k: int = 5) -> list[Retrieved]: ...

class HydeRetriever:
    def __init__(self, ovh, collection, hyde_prompt_template: str): ...
    def search(self, query: str, k: int = 5) -> list[Retrieved]: ...
    # 1) hyde_doc = ovh.chat(hyde_prompt_template.format(q=query))
    # 2) embed hyde_doc → vector
    # 3) collection.query(query_embeddings=[vector], n_results=k)
```

### `pipeline.py`

Responsibility: end-to-end. Accepts a query, returns the answer + the chunks
used (for auditing).

```python
@dataclass
class RagResult:
    answer: str
    retrieved: list[Retrieved]
    debug: dict  # optional: hyde_doc, final prompt, latencies, etc.

class RagPipeline:
    def __init__(self, retriever, ovh, system_prompt: str): ...
    def ask(self, query: str, k: int = 5) -> RagResult: ...
```

Suggested prompt (this is an open decision):
```
System: "You are a technical assistant. Answer using only the CONTEXT. If the
context does not contain the answer, say you don't know. Cite the posts by their
title or slug in parentheses at the end of each statement."

User: "QUESTION: {query}\n\nCONTEXT:\n{joined_chunks}"
```

---

## 7. Data model in ChromaDB

One collection per phase so we can compare without destroying previous results:

| Collection | Strategy |
|---|---|
| `jreypo_naive_v1` | Phase 1 chunks |
| `jreypo_semantic_md_v1` | Phase 2 chunks |

Phase 3 (HyDE) **reuses** `jreypo_semantic_md_v1` — HyDE only changes the
**query**, not the ingested chunks.

Collection configuration:
- Distance: `cosine` (BGE is trained for cosine).
- `embedding_function=None` → we pass the vectors ourselves, rather than having
  Chroma compute them (we don't want it to use its default embedding).

Metadata per document (all keys are strings or numbers in Chroma):
- `post_slug`, `post_title`, `post_date` (ISO), `source_url`
- `chunk_index`, `strategy`, `section_heading` (may be `""` in phase 1)

---

## 8. Phase 1 — Naïve chunking

**Hypothesis**: fixed-size chunks "work reasonably" and provide a baseline.

**Algorithm**: sliding character window with overlap.

Suggested parameters:
- `chunk_chars = 1800` (≈ 450 tokens, comfortably within the model's context).
- `overlap = 200` (≈ 50 tokens; enough to avoid cutting crucial sentences).

**Why characters instead of tokens**: the real tokenizer is that of the
embedding model (BGE / SentencePiece-like). Loading it locally adds a dependency
and weight. For phase 1, characters are a good proxy (≈ 4 chars/token in es/en).

**Predictable failures we're going to observe**:
- Code blocks cut in half.
- One section ends and another begins within the same chunk.
- Orphan headings without their content.
- The golden set should capture these.

---

## 9. Phase 2 — Semantic chunking by Markdown

**Hypothesis**: respecting the structure (H1/H2/H3) preserves topical unity and
raises recall.

**Algorithm**:
1. Parse the body with `markdown-it-py` or another parser that yields typed tokens.
2. Walk the tree grouping content under each heading.
3. If a section exceeds `max_chars = 2400`, sub-chunk it by paragraphs
   (with small overlap).
4. Keep the heading as `section_heading` in metadata.

**Open decisions**:
- What is the minimum heading level we use to split? (My suggestion: H2. H1 is
  usually the title; H3 fragments too much.)
- Include the post title as a prefix of the chunk text to enrich the embedding?
  (Typically improves recall, but pollutes the chunk with redundant info. Worth
  testing.)

**Expected**: Recall@5 ↑ vs phase 1, especially on questions about specific
topics ("how do I configure X in Y?").

---

## 10. Phase 3 — HyDE

**Hypothesis**: queries are **asymmetric** with respect to the corpus (short,
abstract, poorly worded). Embedding them directly produces vectors that don't
live in the same "region" as the chunks (which are long and descriptive).
Generating a hypothetical document brings the query vector closer to the corpus
space.

**Algorithm**:
1. `hyde_doc = chat([{system: "...", user: prompt(query)}])`.
2. `vec = embed(hyde_doc)`.
3. `chunks = collection.query(vec, k)`.

**Suggested prompt** to generate the hyde_doc:

```
You are a technical expert. Write a brief fragment (3-5 sentences) that could be
part of a blog post answering the following question. Write in the same language
as the question. Don't invent proper nouns; if you need an example, leave a
placeholder.

Question: {query}
```

**Open decisions**:
- A single hyde_doc or N=3 with high `temperature` and average the vectors?
  (More diversity → more robust, double the cost.)
- Do we combine HyDE with the original query (RRF, reciprocal rank fusion)?
  Out of scope for the lab, but a good idea to evaluate later.

**Expected**: Recall@5 ↑ vs phase 2 on short/ambiguous queries; may get slightly
worse on very specific queries (the model "hallucinates" details that pull the
vector off course).

---

## 11. Cross-cutting technical decisions

| Topic | Decision | Why |
|---|---|---|
| `chunk_id` | `f"{strategy}:{slug}:{index:04d}"` or `hash(strategy+slug+index)` | Deterministic → reingesting doesn't duplicate. |
| Distance | cosine | BGE is trained for cosine. |
| Embedding dim | 3584 (bge-multi-gemma2) | Fixed by the model. |
| Embed batching | 32–64 texts per call | Tune according to OVH rate limits. |
| Retries | Exponential backoff on 429/5xx, max 5 attempts | Real endpoints fail. |
| Query/corpus language | English corpus; queries may be ES+EN | bge-multi-gemma2 covers both. |
| Token counting | Approximate by characters in phase 1 | Avoids loading a heavy tokenizer. If phase 2/3 need precision, use BGE's HF tokenizer. |
| Logging | `rich` for readable output in notebooks | Not production; readability wins. |
| ID type in Chroma | string | Chroma's API requires it. |
| Text storage | inside Chroma's `documents` field | Avoids a second store for lookup. |
| Ingestion idempotency | `collection.upsert(...)` with deterministic IDs | Re-running the script breaks nothing. |

---

## 12. Evaluation

### Golden set

Proposed format (see `eval/golden_set.yaml`):

```yaml
questions:
  - id: q001
    question: "How did HPVM configure vNICs on HP-UX?"
    expected_slugs:
      - moving-vnics-between-vswitches
    expected_section: null      # or a specific H2/H3
    category: howto             # factual | conceptual | comparison | howto
    notes: "Old HPVM 3.5 post"
```

Approaches to build it (20-30 questions):
- **Factual** (8-10): "what version of X does the post about Y mention?"
- **Howto** (8-10): "how was X done according to the blog?"
- **Conceptual** (4-6): "what does the blog argue about Z?"
- **Comparison** (2-4): "what difference does it mention between A and B?"

### Metrics

Per phase, compute over the whole golden set:
- **Recall@1**: does the first chunk belong to one of the `expected_slugs`?
- **Recall@3**, **Recall@5**.
- **MRR** (Mean Reciprocal Rank): mean of `1/rank` of the first correct chunk.
- (Optional) **Answer quality**: manual 1-5 rubric over 10 queries, evaluating
  faithfulness to the context and completeness.

### Expected final table

```
| Phase               | Recall@1 | Recall@3 | Recall@5 | MRR  |
|---------------------|---------:|---------:|---------:|-----:|
| 1. Naïve            |     0.42 |     0.65 |     0.78 | 0.55 |
| 2. Semantic MD      |     0.55 |     0.78 |     0.88 | 0.68 |
| 3. HyDE (semantic)  |     0.58 |     0.82 |     0.91 | 0.72 |
```

(Made-up figures. What matters is the direction of the deltas.)

---

## 13. Suggested implementation plan

Proposed order. Each step is a short session; commit between steps.

1. **`docker compose up -d chroma`** and verify `curl http://localhost:8000/api/v2/heartbeat`.
2. **venv + install** — `python3.14 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`.
3. **`src/loader.py`** — read 5 posts, print them. Quick test in the REPL.
4. **`src/ovh_client.py`** — `embed(["hello"])` returns 1 vector of 3584. `chat([...])` returns a string.
5. **`scripts/smoke_test.py`** — chain the checks: blog, Chroma, embed, chat.
6. **`src/chunker.py` (naïve)** — `chunk_post_naive` over a sample post, inspect the chunks visually.
7. **`src/ingest.py`** — ingest into `jreypo_naive_v1`. Count how many chunks ended up in Chroma.
8. **`src/retriever.py` (dense)** — run 3 queries by hand, look at the results.
9. **Golden set** — write 20-30 questions in `eval/golden_set.yaml`.
10. **`eval/run_eval.py`** — Recall@k + MRR over phase 1. Save the results.
11. **`src/chunker.py` (semantic_md)** + reingest into `jreypo_semantic_md_v1`.
12. **Re-run eval** → phase 2 vs phase 1.
13. **HyDE** in `retriever.py` (without ingesting anything new).
14. **Re-run eval** → phase 3 vs phase 2.
15. **Notebooks** `01/02/03` — explanation + run + metrics + qualitative observations.
16. **Bonus k8s** — manifests for the homelab.

---

## 14. Open decisions (you resolve these at implementation time)

- BGE's real tokenizer or character approximation? (recommendation: characters in phase 1, decide later).
- Include the post title as a prefix of the chunk text? (recommendation: A/B test in phase 2).
- One OVH base URL for both models or two? (depends on the OVH panel).
- N hyde_docs and average, or a single one? (recommendation: a single one first, N=3 as an improvement if the lab has time).
- `temperature` in HyDE? (recommendation: 0.7 — we want lexical diversity, not determinism).
- Metadata filters in retrieval (by date, by tag)? Useful but out of scope — note it as a follow-up.
- Reranker (cross-encoder) between retrieval and generation? Typically improves recall, out of scope for the lab.

---

## 15. Recommended reading

- Original HyDE: Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (2022).
- BGE family: BAAI documentation on HF (`BAAI/bge-multilingual-gemma2`).
- ChromaDB docs — `HttpClient`, `Collection.upsert`, `Collection.query` sections.
- OpenAI SDK with `base_url`: the canonical example of using the SDK against compatible providers.

---

## Appendix A — Environment variables

Defined in `.env` (not committed). Template in `.env.example`.

| Variable | Example | Notes |
|---|---|---|
| `OVH_AI_API_KEY` | `eyJhbGciOi...` | OVH Bearer token. |
| `OVH_AI_BASE_URL` | `https://.../v1` | No trailing slash. |
| `OVH_EMBEDDING_MODEL` | `bge-multilingual-gemma2` | As it appears in OVH. |
| `OVH_CHAT_MODEL` | `gpt-oss-120b` | Same. |
| `CHROMA_HOST` | `localhost` | In k8s it will be the service name. |
| `CHROMA_PORT` | `8000` | |
| `BLOG_REPO_PATH` | `/Users/jreypo/Documents/Workspace/jreypo.github.io` | Absolute path. |
| `BLOG_POSTS_SUBDIR` | `content/posts` | Relative to `BLOG_REPO_PATH`. |
