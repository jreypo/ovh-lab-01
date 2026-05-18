# Lab 1 — Arquitectura: RAG pipeline sobre jreypo.io

Spec técnica del lab. Sirve de base para que escribas la implementación a mano.
Cada sección define **qué** hay que construir y **por qué**, sin dictar el cómo
en código.

---

## 1. Objetivo

Construir un RAG pipeline funcional sobre el corpus del blog `jreypo.io` y, en
tres fases progresivas, observar cómo cambia la calidad del retrieval cuando
cambiamos la estrategia de chunking y la estrategia de query:

1. **Fase 1**: chunking naïve (tamaño fijo).
2. **Fase 2**: chunking semántico por secciones Markdown.
3. **Fase 3**: HyDE (Hypothetical Document Embeddings).

Lo importante no es llegar al pipeline más bonito sino **medir** la diferencia
entre fases con un golden set propio, y entender por qué cada cambio mueve la
métrica.

---

## 2. Stack

| Capa | Elección | Notas |
|---|---|---|
| Embeddings | OVH AI Endpoints — `bge-multilingual-gemma2` | 3584 dims, contexto 8192 tokens, multilingüe (el blog mezcla ES/EN). €0.01/M tokens. |
| Vector store | ChromaDB (Docker) | Single-node, persistencia en volumen. Cliente Python en modo HTTP. |
| Generación | OVH AI Endpoints — `gpt-oss-120b` | Mismo SDK OpenAI-compat. Usado para respuesta final y para HyDE. |
| Cliente | `openai` SDK con `base_url` apuntando a OVH | OVH expone API compatible OpenAI; un único SDK vale para embed + chat. |
| Runtime dev | Python 3.14 en venv local + Chroma en Docker | Iteración rápida. Chroma fuera de Python evita re-arrancar al cambiar código. |
| Runtime bonus | Kubernetes (homelab) | Al final del lab: deployment de Chroma + Job de ingesta + Deployment de app. |
| Lenguaje | Python ≥ 3.13 | Definido en `pyproject.toml`. |

---

## 3. Vista de componentes

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

## 4. Flujo de datos

**Ingesta** (offline, una vez por fase):

```
posts dir → load_all_posts() → [Post]
            → chunk_*()       → [Chunk]
            → embed(batch)    → [vector]
            → chroma.upsert(ids, texts, metadatas, embeddings)
```

**Query** (online, por pregunta):

```
question
  ├─ (fase 1/2) → embed(question)        ──┐
  └─ (fase 3)   → hyde_doc = chat(prompt)  │
                  → embed(hyde_doc)        │
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
                                      respuesta
```

---

## 5. Estructura del proyecto

Lo que YA está hecho (scaffold):

```
ovh-lab-01/
├── .env.example          ← variables a rellenar
├── .gitignore            ← .env, .venv/, chroma_data/, __pycache__/
├── docker-compose.yml    ← servicio Chroma con volumen persistente
├── Dockerfile            ← imagen Python para fase k8s (CMD por definir)
├── pyproject.toml        ← deps: openai, chromadb, frontmatter, dotenv...
├── src/
│   ├── __init__.py
│   └── config.py         ← carga .env → Settings dataclass
├── eval/
│   └── golden_set.yaml   ← formato vacío esperando preguntas
├── notebooks/            ← vacío
└── k8s/                  ← vacío (bonus)
```

Lo que TÚ vas a crear:

```
src/
├── loader.py             ← lee .md de Hugo → Post
├── chunker.py            ← Post → list[Chunk] (2 estrategias)
├── ovh_client.py         ← wrapper OpenAI-compat (embed + chat)
├── ingest.py             ← orquesta posts → chunks → embed → Chroma
├── retriever.py          ← query → top-k chunks (dense | hyde)
└── pipeline.py           ← end-to-end: query → respuesta

scripts/
└── smoke_test.py         ← chequea blog + Chroma + OVH

eval/
└── run_eval.py           ← Recall@k y MRR sobre golden_set.yaml

notebooks/
├── 01_naive_chunking.ipynb
├── 02_semantic_chunking.ipynb
└── 03_hyde.ipynb
```

---

## 6. Módulos `src/` — contratos sugeridos

Estos son contratos, no implementaciones. Siéntete libre de variar nombres o
firma si en el momento de implementar ves una forma mejor.

### `loader.py`

Responsabilidad: leer los `.md` del directorio `content/posts/` del blog y
producir objetos `Post` con frontmatter parseado.

Contrato:

```python
@dataclass(frozen=True)
class Post:
    slug: str             # extraído del nombre de fichero
    title: str            # de frontmatter
    date: date | None     # de frontmatter o del nombre de fichero
    tags: tuple[str, ...]
    body: str             # markdown sin frontmatter
    source_path: Path

    @property
    def url_path(self) -> str: ...   # /YYYY/MM/DD/slug/

def load_all_posts(posts_dir: Path) -> list[Post]: ...
```

Detalles a resolver al implementar:
- Nombre de fichero del blog: `YYYY-MM-DD-slug.md`. Hay que parsearlo.
- Frontmatter Hugo en YAML entre `---`. Usa `python-frontmatter`.
- Filtrar `_index.md` y ficheros vacíos.
- Tags puede venir como string o lista en YAML.

### `chunker.py`

Responsabilidad: partir un `Post` en `Chunk`s. Dos estrategias separadas, una
por fase.

Contrato común:

```python
@dataclass(frozen=True)
class Chunk:
    chunk_id: str           # determinista; ver §11
    text: str               # contenido del chunk (markdown)
    post_slug: str
    post_title: str
    post_date: str | None   # ISO 8601
    source_url: str         # https://jreypo.io/YYYY/MM/DD/slug/
    chunk_index: int        # orden dentro del post
    strategy: str           # "naive" | "semantic_md"
    section_heading: str | None  # solo en semantic_md

def chunk_post_naive(post: Post, *, chunk_chars: int, overlap: int) -> list[Chunk]: ...
def chunk_post_semantic_md(post: Post, *, max_chars: int) -> list[Chunk]: ...
```

### `ovh_client.py`

Responsabilidad: wrapper fino sobre el SDK `openai` con `base_url` de OVH.

Contrato:

```python
class OVHClient:
    def __init__(self, settings: Settings) -> None: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
    # batching interno si len(texts) > BATCH_SIZE
    # reintentos con backoff exponencial en 429/5xx

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str: ...
```

Detalles a resolver al implementar:
- ¿La URL base de OVH para `bge-multilingual-gemma2` y `gpt-oss-120b` es la
  misma o son distintas? Comprobar en el panel. Si son distintas, dos clientes.
- Batching: 32–64 textos por llamada de embeddings. Mira el límite de OVH.
- Errores transitorios: `tenacity` o un retry sencillo a mano.

### `ingest.py`

Responsabilidad: orquestar posts → chunks → embeddings → Chroma. Idempotente.

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

- Genera chunks con `chunker`.
- Llama a `ovh.embed()` en lotes.
- `collection.upsert(ids=..., documents=..., metadatas=..., embeddings=...)`.
- Idempotencia: usar `Chunk.chunk_id` → reingestar el mismo post no duplica.

### `retriever.py`

Responsabilidad: dada una query, devolver top-k chunks. Dos estrategias.

Contrato:

```python
@dataclass
class Retrieved:
    chunk_id: str
    text: str
    metadata: dict
    score: float           # similaridad / 1 - distance

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

Responsabilidad: end-to-end. Acepta una query, devuelve respuesta + chunks
usados (para auditoría).

```python
@dataclass
class RagResult:
    answer: str
    retrieved: list[Retrieved]
    debug: dict  # opcional: hyde_doc, prompt final, latencias, etc.

class RagPipeline:
    def __init__(self, retriever, ovh, system_prompt: str): ...
    def ask(self, query: str, k: int = 5) -> RagResult: ...
```

Prompt sugerido (es decisión abierta):
```
Sistema: "Eres un asistente técnico. Responde usando solo el CONTEXTO. Si el
contexto no contiene la respuesta, di que no lo sabes. Cita los posts por su
título o slug entre paréntesis al final de cada afirmación."

Usuario: "PREGUNTA: {query}\n\nCONTEXTO:\n{joined_chunks}"
```

---

## 7. Modelo de datos en ChromaDB

Una colección por fase para poder comparar sin destruir resultados anteriores:

| Colección | Estrategia |
|---|---|
| `jreypo_naive_v1` | Chunks fase 1 |
| `jreypo_semantic_md_v1` | Chunks fase 2 |

Fase 3 (HyDE) **reusa** `jreypo_semantic_md_v1` — HyDE solo cambia la **query**,
no los chunks ingestados.

Configuración de la colección:
- Distance: `cosine` (BGE entrena para cosine).
- `embedding_function=None` → pasamos los vectores nosotros, no que Chroma los
  calcule (no queremos que use su embedding por defecto).

Metadata por documento (todos los keys son strings o números en Chroma):
- `post_slug`, `post_title`, `post_date` (ISO), `source_url`
- `chunk_index`, `strategy`, `section_heading` (puede ser `""` en fase 1)

---

## 8. Fase 1 — Chunking naïve

**Hipótesis**: chunks de tamaño fijo "funcionan razonable" y dan baseline.

**Algoritmo**: ventana deslizante por caracteres con solape.

Parámetros sugeridos:
- `chunk_chars = 1800` (≈ 450 tokens, holgado dentro del contexto del modelo).
- `overlap = 200` (≈ 50 tokens; suficiente para no cortar oraciones cruciales).

**Por qué caracteres en vez de tokens**: el tokenizer real es el del modelo de
embeddings (BGE / SentencePiece-like). Cargarlo localmente añade dependencia y
peso. Para fase 1, caracteres es buen proxy (≈ 4 chars/token en es/en).

**Fallos predecibles que vamos a observar**:
- Bloques de código cortados a mitad.
- Una sección termina y otra empieza dentro del mismo chunk.
- Headings sueltos sin su contenido.
- El golden set debería capturarlos.

---

## 9. Fase 2 — Chunking semántico por Markdown

**Hipótesis**: respetar la estructura (H1/H2/H3) preserva unidad temática y
sube recall.

**Algoritmo**:
1. Parsear el body con `markdown-it-py` u otro parser que dé tokens con tipo.
2. Recorrer el árbol agrupando contenido bajo cada heading.
3. Si una sección supera `max_chars = 2400`, sub-chunkearla por párrafos
   (con overlap pequeño).
4. Conservar el heading como `section_heading` en metadata.

**Decisiones abiertas**:
- ¿Cuál es el nivel mínimo de heading que usamos para cortar? (Mi sugerencia:
  H2. H1 suele ser el título; H3 fragmenta demasiado.)
- ¿Incluir el título del post como prefijo del texto del chunk para enriquecer
  el embedding? (Mejora típicamente recall, pero contamina el chunk con info
  redundante. A probar.)

**Esperable**: Recall@5 ↑ vs fase 1, sobre todo en preguntas sobre temas
concretos ("¿cómo configuro X en Y?").

---

## 10. Fase 3 — HyDE

**Hipótesis**: las queries son **asimétricas** respecto al corpus (cortas,
abstractas, mal redactadas). Embeddearlas directamente lleva a vectores que no
viven en la misma "región" que los chunks (que son largos y descriptivos).
Generar un documento hipotético acerca el vector de la query al espacio del
corpus.

**Algoritmo**:
1. `hyde_doc = chat([{system: "...", user: prompt(query)}])`.
2. `vec = embed(hyde_doc)`.
3. `chunks = collection.query(vec, k)`.

**Prompt sugerido** para generar el hyde_doc:

```
Eres un experto técnico. Escribe un fragmento breve (3-5 frases) que podría
formar parte de un post de blog respondiendo a la siguiente pregunta. Escribe
en el mismo idioma que la pregunta. No inventes nombres propios; si necesitas
un ejemplo, deja un placeholder.

Pregunta: {query}
```

**Decisiones abiertas**:
- ¿Un solo hyde_doc o N=3 con `temperature` alta y promediamos vectores?
  (Más diversidad → más robusto, doble coste.)
- ¿Combinamos HyDE con la query original (RRF, reciprocal rank fusion)? Fuera
  del alcance del lab, pero buena idea para evaluar luego.

**Esperable**: Recall@5 ↑ vs fase 2 en queries cortas/ambiguas; puede empeorar
ligeramente en queries muy específicas (el modelo "alucina" detalles que
desvían el vector).

---

## 11. Decisiones técnicas transversales

| Tema | Decisión | Por qué |
|---|---|---|
| `chunk_id` | `f"{strategy}:{slug}:{index:04d}"` o `hash(strategy+slug+index)` | Determinista → reingestar no duplica. |
| Distancia | cosine | BGE entrena para cosine. |
| Dim. embedding | 3584 (bge-multi-gemma2) | Fijado por el modelo. |
| Batching embed | 32–64 textos por llamada | Ajustar según rate limits de OVH. |
| Reintentos | Backoff exponencial en 429/5xx, máx 5 intentos | Endpoints reales fallan. |
| Lenguaje query/corpus | Multilingüe ES+EN | bge-multi-gemma2 lo cubre. |
| Counting tokens | Aproximar por caracteres en fase 1 | Evita cargar tokenizer pesado. Si en fase 2/3 hace falta precisión, usar el tokenizer de HF de BGE. |
| Logging | `rich` para output legible en notebooks | No es producción; legibilidad gana. |
| Tipo de IDs en Chroma | string | API de Chroma lo requiere. |
| Almacenamiento del texto | dentro del campo `documents` de Chroma | Evita una segunda store para lookup. |
| Idempotencia ingesta | `collection.upsert(...)` con IDs deterministas | Re-correr el script no rompe nada. |

---

## 12. Evaluación

### Golden set

Formato propuesto (ver `eval/golden_set.yaml`):

```yaml
questions:
  - id: q001
    question: "¿Cómo configuraba HPVM las vNICs en HP-UX?"
    expected_slugs:
      - moving-vnics-between-vswitches
    expected_section: null      # o un H2/H3 concreto
    category: howto             # factual | conceptual | comparativa | howto
    notes: "Post antiguo de HPVM 3.5"
```

Aproximaciones para construirlo (20-30 preguntas):
- **Factual** (8-10): "¿qué versión de X menciona el post sobre Y?"
- **Howto** (8-10): "¿cómo se hacía X según el blog?"
- **Conceptual** (4-6): "¿qué argumenta el blog sobre Z?"
- **Comparativa** (2-4): "¿qué diferencia menciona entre A y B?"

### Métricas

Por fase, calcular sobre todo el golden set:
- **Recall@1**: ¿el primer chunk pertenece a uno de los `expected_slugs`?
- **Recall@3**, **Recall@5**.
- **MRR** (Mean Reciprocal Rank): media de `1/rank` del primer chunk correcto.
- (Opcional) **Calidad de respuesta**: rúbrica 1-5 manual sobre 10 queries,
  evaluando fidelidad al contexto y completitud.

### Tabla final esperada

```
| Fase                | Recall@1 | Recall@3 | Recall@5 | MRR  |
|---------------------|---------:|---------:|---------:|-----:|
| 1. Naïve            |     0.42 |     0.65 |     0.78 | 0.55 |
| 2. Semantic MD      |     0.55 |     0.78 |     0.88 | 0.68 |
| 3. HyDE (semantic)  |     0.58 |     0.82 |     0.91 | 0.72 |
```

(Cifras inventadas. Lo importante es la dirección de los deltas.)

---

## 13. Plan de implementación sugerido

Orden propuesto. Cada paso es una sesión corta; commitea entre pasos.

1. **`docker compose up -d chroma`** y verifica `curl http://localhost:8000/api/v2/heartbeat`.
2. **venv + install** — `python3.14 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`.
3. **`src/loader.py`** — leer 5 posts, imprimirlos. Test rápido en REPL.
4. **`src/ovh_client.py`** — método `embed(["hola"])` devuelve 1 vector de 3584. Método `chat([...])` devuelve string.
5. **`scripts/smoke_test.py`** — encadena los chequeos: blog, Chroma, embed, chat.
6. **`src/chunker.py` (naïve)** — `chunk_post_naive` sobre un post de ejemplo, inspecciona los chunks visualmente.
7. **`src/ingest.py`** — ingesta a `jreypo_naive_v1`. Cuenta cuántos chunks acabaron en Chroma.
8. **`src/retriever.py` (dense)** — lanza 3 queries a mano, mira resultados.
9. **Golden set** — escribe 20-30 preguntas en `eval/golden_set.yaml`.
10. **`eval/run_eval.py`** — Recall@k + MRR sobre fase 1. Guarda resultados.
11. **`src/chunker.py` (semantic_md)** + reingest a `jreypo_semantic_md_v1`.
12. **Re-correr eval** → fase 2 vs fase 1.
13. **HyDE** en `retriever.py` (sin ingestar nada nuevo).
14. **Re-correr eval** → fase 3 vs fase 2.
15. **Notebooks** `01/02/03` — explicación + run + métricas + observaciones cualitativas.
16. **Bonus k8s** — manifests para el homelab.

---

## 14. Decisiones abiertas (las resuelves tú al implementar)

- ¿Tokenizer real de BGE o aproximación por caracteres? (recomendación: caracteres en fase 1, decidir luego).
- ¿Incluir título del post como prefijo del texto del chunk? (recomendación: probar A/B en fase 2).
- ¿Una URL base de OVH para ambos modelos o dos? (depende del panel de OVH).
- ¿N hyde_docs y promediamos, o uno solo? (recomendación: uno solo primero, N=3 como mejora si el lab da tiempo).
- ¿`temperature` en HyDE? (recomendación: 0.7 — queremos diversidad léxica, no determinismo).
- ¿Filtros de metadata en el retrieval (por fecha, por tag)? Útil pero fuera del alcance — anótalo como follow-up.
- ¿Reranker (cross-encoder) entre retrieval y generación? Mejora típicamente recall, fuera del alcance del lab.

---

## 15. Lecturas recomendadas

- HyDE original: Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (2022).
- BGE family: BAAI documentation en HF (`BAAI/bge-multilingual-gemma2`).
- ChromaDB docs — secciones de `HttpClient`, `Collection.upsert`, `Collection.query`.
- OpenAI SDK con `base_url`: ejemplo canónico de usar el SDK contra proveedores compatibles.

---

## Apéndice A — Variables de entorno

Definidas en `.env` (no commiteado). Plantilla en `.env.example`.

| Variable | Ejemplo | Notas |
|---|---|---|
| `OVH_AI_API_KEY` | `eyJhbGciOi...` | Bearer token de OVH. |
| `OVH_AI_BASE_URL` | `https://.../v1` | Sin trailing slash. |
| `OVH_EMBEDDING_MODEL` | `bge-multilingual-gemma2` | Tal como aparece en OVH. |
| `OVH_CHAT_MODEL` | `gpt-oss-120b` | Idem. |
| `CHROMA_HOST` | `localhost` | En k8s será el service name. |
| `CHROMA_PORT` | `8000` | |
| `BLOG_REPO_PATH` | `/Users/jreypo/Documents/Workspace/jreypo.github.io` | Path absoluto. |
| `BLOG_POSTS_SUBDIR` | `content/posts` | Relativo a `BLOG_REPO_PATH`. |
