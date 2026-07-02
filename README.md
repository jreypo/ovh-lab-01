# ovh-lab-01 — RAG pipeline over jreypo.io

A lab to build, **by hand**, a RAG (Retrieval-Augmented Generation) pipeline
over the corpus of the [`jreypo.io`](https://jreypo.io) blog, using
[OVH AI Endpoints](https://endpoints.ai.cloud.ovh.net/) for embeddings and
generation, and [ChromaDB](https://www.trychroma.com/) as the vector store.

The goal is not the "prettiest" pipeline, but to **measure** how retrieval
quality changes across three progressive phases and to understand *why* each
change moves the metric:

1. **Phase 1 — Naïve chunking**: fixed-size character window with overlap. Baseline.
2. **Phase 2 — Semantic chunking**: splits that respect the Markdown structure (H2/H3).
3. **Phase 3 — HyDE**: *Hypothetical Document Embeddings* — a hypothetical document
   is generated from the query and that document is embedded instead of the raw query.

> The full technical spec (per-module contracts, algorithms, open decisions)
> lives in [`ARCHITECTURE.md`](./ARCHITECTURE.md). This README is the entry
> point; that document is the reference.

---

## Stack

| Layer | Choice |
|---|---|
| Embeddings | OVH AI Endpoints — `bge-multilingual-gemma2` (3584 dims, multilingual — English corpus, queries may be ES/EN) |
| Generation | OVH AI Endpoints — `gpt-oss-120b` (final answer + HyDE generation) |
| Client | `openai` SDK with `base_url` pointing to OVH (OpenAI-compatible API) |
| Vector store | ChromaDB in Docker (HTTP, volume persistence) |
| Dev runtime | Python ≥ 3.13 in a local venv + Chroma in a container |
| Bonus runtime | Kubernetes (homelab) |

---

## Repository structure

```
ovh-lab-01/
├── ARCHITECTURE.md       Detailed technical spec (READ it first)
├── README.md             This file
├── .env.example          Environment variables template
├── docker-compose.yml    ChromaDB service with persistent volume
├── Dockerfile            Python image for the k8s phase
├── pyproject.toml        Dependencies and ruff config
├── src/
│   ├── config.py         Loads .env → Settings (DONE)
│   ├── loader.py         Hugo .md → Post                       (to write)
│   ├── chunker.py        Post → list[Chunk] (naive | semantic_md)  (to write)
│   ├── ovh_client.py     OpenAI-compat wrapper: embed() + chat()    (to write)
│   ├── ingest.py         posts → chunks → embed → Chroma        (to write)
│   ├── retriever.py      query → top-k chunks (dense | hyde)    (to write)
│   └── pipeline.py       end-to-end: query → answer            (to write)
├── scripts/
│   └── smoke_test.py     Checks blog + Chroma + OVH             (to write)
├── eval/
│   ├── golden_set.yaml   Evaluation questions (to be filled in)
│   └── run_eval.py       Recall@k + MRR                         (to write)
├── notebooks/            01_naive / 02_semantic / 03_hyde       (to write)
└── k8s/                  Manifests for the homelab (bonus)      (to write)
```

The scaffold (`config.py`, `docker-compose.yml`, `Dockerfile`, `pyproject.toml`,
`.env.example`, golden set format) is done. The rest of `src/` is implemented
following the contracts in [`ARCHITECTURE.md`](./ARCHITECTURE.md) §6.

---

## Architecture (overview)

```
Hugo blog (.md)  ──► loader.py ──► chunker.py ──┐
                                                 │ texts
                                                 ▼
                                          ovh_client.embed() ──► OVH AI Endpoints
                                                 │ vectors
                                                 ▼
                                          ingest.py ──► ChromaDB (1 collection/phase)
                                                            ▲
   query ──► retriever.py (dense | hyde) ──► query top-k ───┘
                  │
                  ▼
            pipeline.py ──► ovh_client.chat() ──► answer
```

**Chroma collections** (one per phase, to compare without destroying results):

| Collection | Strategy |
|---|---|
| `jreypo_naive_v1` | Phase 1 |
| `jreypo_semantic_md_v1` | Phase 2 (HyDE reuses it: only the query changes) |

Distance `cosine`, dimension fixed by the model (3584). Details in
[`ARCHITECTURE.md`](./ARCHITECTURE.md) §7 and §11.

---

## Getting started

### 1. Configure environment variables

```bash
cp .env.example .env
# Fill in OVH_AI_API_KEY, OVH_AI_BASE_URL and adjust BLOG_REPO_PATH
```

| Variable | Notes |
|---|---|
| `OVH_AI_API_KEY` | Bearer token for OVH AI Endpoints |
| `OVH_AI_BASE_URL` | Base URL `…/v1`, no trailing slash |
| `OVH_EMBEDDING_MODEL` | `bge-multilingual-gemma2` |
| `OVH_CHAT_MODEL` | `gpt-oss-120b` |
| `CHROMA_HOST` / `CHROMA_PORT` | `localhost` / `8000` (matches docker-compose) |
| `BLOG_REPO_PATH` | Absolute path to the Hugo blog repo |
| `BLOG_POSTS_SUBDIR` | `content/posts` (relative to `BLOG_REPO_PATH`) |

### 2. Start ChromaDB

```bash
docker compose up -d chroma
curl http://localhost:8000/api/v2/heartbeat   # must return 200
```

### 3. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Lab workflow

The step-by-step implementation plan is in
[`ARCHITECTURE.md`](./ARCHITECTURE.md) §13. In short:

1. `docker compose up -d chroma` + verify the heartbeat.
2. Implement `loader.py` and `ovh_client.py`; validate with `scripts/smoke_test.py`.
3. Phase 1: `chunker.py` (naïve) → `ingest.py` to `jreypo_naive_v1` → `retriever.py` (dense).
4. Write 20-30 questions in `eval/golden_set.yaml` and run `eval/run_eval.py`.
5. Phase 2: `chunker.py` (semantic_md) → reingest to `jreypo_semantic_md_v1` → re-evaluate.
6. Phase 3: HyDE in `retriever.py` (no reingest) → re-evaluate.
7. Notebooks `01/02/03` with explanation, runs, and observations.
8. Bonus: k8s manifests for the homelab.

---

## Evaluation

Each phase is measured over the same golden set (`eval/golden_set.yaml`) with:

- **Recall@1 / @3 / @5** — does a chunk from an `expected_slug` appear in the top-k?
- **MRR** — Mean Reciprocal Rank of the first correct chunk.

What matters is the **direction of the deltas** between phases, not the absolute
values. Expected shape of the final table:

```
| Phase               | Recall@1 | Recall@3 | Recall@5 | MRR  |
|---------------------|---------:|---------:|---------:|-----:|
| 1. Naïve            |        … |        … |        … |    … |
| 2. Semantic MD      |        … |        … |        … |    … |
| 3. HyDE (semantic)  |        … |        … |        … |    … |
```

Hypothesis: semantic raises recall vs naïve (it preserves topical unity); HyDE
raises it on short/ambiguous queries (it brings the query vector closer to the
corpus space). Full reasoning in [`ARCHITECTURE.md`](./ARCHITECTURE.md) §8–§10.

---

## Notes

- `.env`, `.venv/` and `chroma_data/` are in `.gitignore`; they are not committed.
- Ingestion is idempotent: deterministic `chunk_id` + `collection.upsert(...)`,
  so re-running the script does not duplicate.
- Lint with `ruff` (config in `pyproject.toml`).

---

## References

- HyDE: Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (2022).
- BGE: `BAAI/bge-multilingual-gemma2` (Hugging Face).
- ChromaDB docs — `HttpClient`, `Collection.upsert`, `Collection.query`.
