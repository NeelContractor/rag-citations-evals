.PHONY: setup lint test ingest

UV ?= uv
DOCS ?= data/docs
DB ?= .lancedb
STRATEGY ?= heading
CHUNK_SIZE ?= 800
CHUNK_OVERLAP ?= 120

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
