.PHONY: setup lint test ingest serve golden eval

UV ?= uv
DOCS ?= data/docs
DB ?= .lancedb
STRATEGY ?= heading
CHUNK_SIZE ?= 800
CHUNK_OVERLAP ?= 120
PORT ?= 8000
GOLDEN_OUT ?= data/evals/golden_draft.jsonl
GOLDEN_COUNT ?= 30
GOLDEN_UNANSWERABLE ?= 5
GOLDEN_MODE ?= auto
GOLDEN_MODEL ?=
EVAL_GOLDEN ?= data/evals/golden.jsonl
EVAL_CHUNK_SIZES ?= 400,800,1200
EVAL_OVERLAP ?= 100
EVAL_STRATEGY ?= heading
EVAL_TOP_K ?= 5
EVAL_OUTPUT ?= eval_results
EVAL_FLAGS ?=

setup:
	mkdir -p $(DOCS)
	$(UV) sync --all-groups
	$(UV) run pre-commit install

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

test:
	$(UV) run pytest

ingest:
	$(UV) run python -m rag_eval.ingest \
		--docs $(DOCS) \
		--db $(DB) \
		--strategy $(STRATEGY) \
		--chunk-size $(CHUNK_SIZE) \
		--chunk-overlap $(CHUNK_OVERLAP)

serve:
	$(UV) run uvicorn rag_eval.api.app:app --host 127.0.0.1 --port $(PORT) --reload

golden: ingest
	$(UV) run python -m rag_eval.eval.generate_golden \
		--db $(DB) \
		--out $(GOLDEN_OUT) \
		--count $(GOLDEN_COUNT) \
		--unanswerable $(GOLDEN_UNANSWERABLE) \
		--mode $(GOLDEN_MODE) \
		--force \
		$(if $(GOLDEN_MODEL),--model $(GOLDEN_MODEL),)

eval:
	@test -f $(EVAL_GOLDEN) || (echo "Missing golden set: $(EVAL_GOLDEN)"; echo "Run 'make golden', review the draft, then copy it to $(EVAL_GOLDEN)."; exit 1)
	$(UV) run python -m rag_eval.eval \
		--docs $(DOCS) \
		--golden $(EVAL_GOLDEN) \
		--db $(DB) \
		--chunk-sizes $(EVAL_CHUNK_SIZES) \
		--overlap $(EVAL_OVERLAP) \
		--strategy $(EVAL_STRATEGY) \
		--top-k $(EVAL_TOP_K) \
		--output $(EVAL_OUTPUT) \
		$(EVAL_FLAGS)
