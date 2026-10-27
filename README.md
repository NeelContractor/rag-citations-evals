# RAG Citations & Eval

A small, local-first RAG system that answers questions from your documents with
verified source citations, refuses when the retrieved context is insufficient, and
includes an evaluation suite for measuring retrieval quality, citation quality, and
answer faithfulness.

The retrieval stack runs locally on CPU. Generation and optional LLM judging use
OpenRouter.

## Features

- PDF and Markdown ingestion with fixed-size or heading-aware chunking
- Local embeddings with `BAAI/bge-small-en-v1.5`
- LanceDB vector store
- Vector search, in-house BM25, reciprocal-rank fusion, and optional late-interaction reranking
- Strict JSON generation with quote-level citation verification
- Automatic refusal when no retrieved citation supports an answer
- FastAPI endpoint and single-page browser UI
- Golden-set generation helper and a 12-configuration evaluation matrix
- Retrieval-only evaluation mode that does not require an API key

## Architecture

```text
data/docs/*.pdf|*.md
        |
        v
 loaders -> chunking -> fastembed embeddings -> LanceDB
                                              |
question -> query embedding -> vector search -+-> BM25 -> RRF
                                              |
                                     optional local reranker
                                              |
                                        top-k chunks
                                              |
                                   OpenRouter generation
                                              |
                              strict JSON + citation checks
                                              |
                              answer with sources, or refusal
```

Generation only sees the retrieved chunks. Every returned quote is checked against
the cited chunk; invalid citations are dropped. If no valid citation remains and
citations are required, the system returns a refusal instead of an unsupported answer.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Internet access for OpenRouter generation
- A few hundred MB of local model downloads on first use
- OpenRouter API key for asking questions and running full evaluations

Ingestion, retrieval-only evaluation, and heuristic golden-set generation do not
call an LLM.

## Setup

```bash
cd RAG_Eval

cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY

make setup
```

`make setup` installs dependencies, creates `data/docs`, and installs the pre-commit
hooks.

Add PDF, `.md`, or `.markdown` files under `data/docs`, including subdirectories,
then build the index:

```bash
make ingest
```

The default index is written to `.lancedb/chunks` using heading-aware chunking with
an 800-character target and 120-character overlap.

## Configuration

`.env` is loaded automatically by the CLI, API, and evaluation commands.

```dotenv
OPENROUTER_API_KEY=your-key
OPENROUTER_MODEL=google/gemma-4-26b-a4b-it:free
```

- `OPENROUTER_API_KEY` is required for generation and LLM judging.
- `OPENROUTER_MODEL` is optional; the default is the free-tier model above.
- `.env` is gitignored. Never commit API keys.

Other defaults live in `src/rag_eval/config.py`:

| Setting | Default |
| --- | --- |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Reranker | `answerdotai/answerai-colbert-small-v1` |
| Vector store | `.lancedb/chunks` |
| Retrieval `top_k` | 5 |
| Retrieval `candidate_k` | 20 |
| Chunking | heading-aware, 800 chars, 120 overlap |
| Generation | temperature 0, max 800 tokens, 1 retry |

## Ingesting Documents

```bash
make ingest

# or explicitly:
uv run python -m rag_eval.ingest \
  --docs data/docs \
  --db .lancedb \
  --strategy heading \
  --chunk-size 800 \
  --chunk-overlap 120
```

Use `--strategy fixed` for plain fixed-size chunking. PDFs always use fixed-size
chunking and retain page numbers.

The index is replaced when `make ingest` runs. Re-run it after changing documents
or chunking settings.

## Asking Questions

CLI:

```bash
uv run python -m rag_eval.ask "What is retrieval-augmented generation?"

# with hybrid retrieval and reranking:
uv run python -m rag_eval.ask "Why cite sources?" --hybrid --rerank

# machine-readable output:
uv run python -m rag_eval.ask "Why cite sources?" --json
```

Options include `--top-k`, `--candidate-k`, `--hybrid`, `--rerank`, `--model`,
`--db`, and `--table`.

Example output:

```text
RAG combines a retriever with a language model...

Sources:
- sample.md: "Retrieval-augmented generation (RAG) combines..."
```

If the model or citation checks cannot support an answer, the CLI prints a refusal
and the reason.

### API and browser UI

```bash
make serve
```

Open <http://127.0.0.1:8000> for the UI.

- `GET /health` returns `{"status": "ok"}`.
- `GET /` serves the single-page UI.
- `POST /ask` accepts:

```json
{
  "question": "What is RAG?",
  "top_k": 5,
  "hybrid": true,
  "rerank": false
}
```

```bash
curl -s http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is RAG?","top_k":5,"hybrid":true,"rerank":true}'
```

The response contains the answer, refusal status, model, latency, and cited sources
with the exact quote and highlighted HTML for display.

The API has no authentication. Keep it bound to `127.0.0.1` and do not expose it
publicly without adding authentication and request limits.

## Evaluation

The evaluation suite compares chunking sizes and retrieval configurations using a
reviewed golden set.

### 1. Create a draft golden set

```bash
# Uses OPENROUTER_MODEL when a key is available; otherwise falls back to heuristics.
make golden

# Force heuristic generation:
make golden GOLDEN_MODE=heuristic

# Customize counts:
make golden GOLDEN_COUNT=40 GOLDEN_UNANSWERABLE=8
```

This runs ingestion first and writes `data/evals/golden_draft.jsonl`. The default is
30 answerable and 5 unanswerable questions. The number of answerable questions is
limited by how many useful chunks exist in the corpus.

Review every draft item: questions, reference answers, and evidence quotes. Then
copy the reviewed file:

```bash
cp data/evals/golden_draft.jsonl data/evals/golden.jsonl
```

Golden files are local and gitignored. They are not committed with the project.

### Golden JSONL schema

```json
{
  "id": "q001",
  "question": "What is retrieval-augmented generation?",
  "reference_answer": "RAG combines a document retriever with a language model.",
  "answerable": true,
  "evidence": [
    {
      "doc": "sample.md",
      "page": null,
      "quote": "Retrieval-augmented generation (RAG) combines a document retriever with a language model."
    }
  ],
  "source_chunk_ids": ["sample-md-p0-0000-abc123"],
  "notes": ""
}
```

- `evidence[].quote` must be text from the source document.
- For PDFs, set `page` to the 1-based page number.
- `source_chunk_ids` is optional and only documents the chunk IDs from the default
  index; evaluation remaps evidence to each chunk size.
- Answerable items require at least one evidence span.
- Unanswerable items use `"answerable": false` and an empty `evidence` list.

Evidence mapping first looks for an exact case-insensitive, whitespace-normalized
quote. If that fails, it falls back to token coverage with an 80% threshold. This
allows the same golden item to be evaluated across 400, 800, and 1200-character
chunkings.

### 2. Run the matrix

Retrieval-only smoke test, no API key required:

```bash
make eval EVAL_FLAGS="--retrieval-only"
```

Small smoke run:

```bash
make eval EVAL_FLAGS="--retrieval-only --limit 5 --chunk-sizes 800"
```

Full evaluation, including generation and faithfulness judging:

```bash
make eval
```

By default this runs 12 configurations:

- Chunk sizes: 400, 800, 1200 characters
- Overlap: 100 characters
- Hybrid retrieval: off/on
- Reranking: off/on

Common overrides:

```bash
make eval EVAL_CHUNK_SIZES=400,800 EVAL_TOP_K=3
make eval EVAL_FLAGS="--reuse-index"
make eval EVAL_FLAGS="--judge-model google/gemma-4-26b-a4b-it:free"
```

`--reuse-index` skips re-ingestion when the evaluation tables already exist. Use it
only when the documents and chunking settings have not changed.

Results are printed as a comparison table and saved to
`eval_results/eval_<timestamp>.json` and `eval_results/latest.json`.

### Metrics

| Metric | Meaning |
| --- | --- |
| `recall@k` | Fraction of gold chunks present in the top-k retrieved chunks |
| `mrr` | Reciprocal rank of the first gold chunk |
| `faithfulness` | LLM-judge score for claims supported by the retrieved context |
| `citation_precision` | Share of cited/invalid citations that point to gold chunks |
| `citation_recall` | Share of gold chunks cited in the answer |
| `citation_validity` | Share of returned citations whose quotes were verified |
| `refusal_rate_unanswerable` | Share of unanswerable questions correctly refused |
| `answer_rate_answerable` | Share of answerable questions that were answered |
| `generation_errors` | Failed generations |
| `judge_errors` | Failed faithfulness judgements |

Citation precision counts invalid citations in the denominator, so unsupported or
hallucinated quotes reduce the score. Faithfulness is averaged over answerable
questions that were actually answered and successfully judged; refusals are measured
separately.

Full evaluations make many LLM calls. Free-tier rate limits and latency can make a
30-question, 12-configuration run slow. Start with `--retrieval-only`, then use
`--limit` for a smaller generation smoke test.

## Development

```bash
make lint          # ruff check + format check
make test          # pytest
uv run pre-commit run --all-files
```

The test suite covers chunking, ingestion, retrieval fusion/reranking, citation
verification, generation parsing, the API, golden mapping, metrics, the judge, and
the evaluation runner.

## Project Layout

```text
src/rag_eval/
  config.py              Configuration models and .env loading
  ask.py                 Ask CLI
  ingest/                Loaders, chunking, embeddings, LanceDB store
  retrieval/             Vector search, BM25/RRF, reranker
  generation/            Prompts, OpenRouter client, citation verification
  api/                   FastAPI app, response models, UI
  eval/                  Golden schema, generation, metrics, judge, matrix runner
tests/                   Unit and API tests
data/docs/               Local documents (gitignored)
data/evals/              Local golden sets (gitignored)
eval_results/            Evaluation output (gitignored)
.lancedb/                Local vector store (gitignored)
```

## Design Notes

- **Local retrieval, remote generation.** Embeddings and reranking run on CPU; only
  retrieved chunks and the question are sent to OpenRouter.
- **Quotes are verified, not trusted.** Model-provided citations are checked against
  the exact retrieved chunk text. Invalid citations are dropped and counted.
- **Refusal is a first-class result.** Missing context or no valid citations produces
  a refusal rather than an unsupported answer.
- **BM25 is in-house.** It avoids an extra dependency and keeps the retrieval logic
  inspectable.
- **Chunk-size-independent gold data.** Evidence spans, not chunk IDs, are the source
  of truth for evaluation.
- **No hidden data collection.** The project has no telemetry. Documents, indexes,
  golden sets, and results stay local and are gitignored.

## Limitations

- Scanned PDFs are not OCRed; pypdf only extracts embedded text.
- The default embedding model is optimized for English text.
- LLM-as-judge scores are approximate and can disagree with human judgment.
- Heuristic golden drafts are intentionally simple and need review.
- There is no streaming, authentication, multi-user isolation, or rate limiting.
- The default free-tier model may change or be unavailable; set
  `OPENROUTER_MODEL` to another OpenRouter model if needed.

## Troubleshooting

**`OPENROUTER_API_KEY is not set`**
Set it in `.env` or the environment. Use `--retrieval-only` for evaluation without it.

**`Vector store not found` or `Table 'chunks' not found`**
Run `make ingest` and confirm documents exist under `data/docs`.

**`Golden set not found`**
Run `make golden`, review `data/evals/golden_draft.jsonl`, and copy it to
`data/evals/golden.jsonl`.

**First run is slow**
The embedding and reranker models download on first use and are cached under
`.cache/fastembed`.

**OpenRouter 429 or 5xx errors**
Free-tier rate limits can be tight. Retry later, use a smaller `--limit`, or choose
another model with `--model`/`OPENROUTER_MODEL`.
