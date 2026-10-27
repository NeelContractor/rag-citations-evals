.PHONY: setup lint test ingest serve

UV ?= uv
DOCS ?= data/docs
DB ?= .lancedb
STRATEGY ?= heading
CHUNK_SIZE ?= 800
CHUNK_OVERLAP ?= 120
PORT ?= 8000

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
