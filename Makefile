# Codebase Agent RAG Makefile
# Shortcuts for Python-based build system

.PHONY: help build test clean deps syntax imports ollama debug interactive

# Virtual environment activation
VENV_ACTIVATE = . venv_rag/bin/activate &&

# Default target
help:
	@echo "Codebase Agent RAG Build System"
	@echo ""
	@echo "Available commands:"
	@echo "  make build        - Complete build process"
	@echo "  make test         - Run all tests"
	@echo "  make test-rag     - Run RAG decision tests"
	@echo "  make test-llm     - Run LLM judgment tests"
	@echo "  make test-improved - Run improved RAG tests"
	@echo "  make syntax       - Syntax check"
	@echo "  make imports      - Import test"
	@echo "  make clean        - Clean temporary files"
	@echo "  make deps         - Install dependencies"
	@echo "  make ollama       - Check Ollama service"
	@echo "  make interactive  - Interactive test"
	@echo "  make debug        - Debug specific query"
	@echo "  make help         - Show help"

# Complete build process
build:
	$(VENV_ACTIVATE) python codebase_agent_rag.py index

# Run all tests
test:
	@echo "Running RAG decision tests..."
	$(VENV_ACTIVATE) python codebase_agent_rag.py test-rag-decision "How to implement user authentication feature?"
	@echo "Running LLM judgment tests..."
	$(VENV_ACTIVATE) python codebase_agent_rag.py debug-llm-judgment "What are the main features of this project?"

# Run specific tests
test-rag:
	@echo "Running RAG decision tests..."
	$(VENV_ACTIVATE) python codebase_agent_rag.py test-rag-decision "How to implement user authentication feature?"

test-llm:
	@echo "Running LLM judgment tests..."
	$(VENV_ACTIVATE) python codebase_agent_rag.py debug-llm-judgment "What are the main features of this project?"

test-improved:
	@echo "Running improved RAG tests..."
	$(VENV_ACTIVATE) python codebase_agent_rag.py test-rag-decision "How to optimize code performance?"

# Syntax check
syntax:
	$(VENV_ACTIVATE) python -m py_compile codebase_agent_rag.py
	$(VENV_ACTIVATE) python -m py_compile debug_utilities.py

# Import test
imports:
	$(VENV_ACTIVATE) python -c "import codebase_agent_rag; import debug_utilities; print('✅ All imports successful')"

# Clean temporary files
clean:
	rm -rf __pycache__
	rm -rf .codebase_index
	rm -rf *.pyc
	rm -rf .pytest_cache
	rm -rf .coverage

# Install dependencies
deps:
	$(VENV_ACTIVATE) pip install -r requirements_rag.txt

# Check Ollama service
ollama:
	@echo "Checking Ollama service status..."
	@curl -s http://localhost:11434/api/tags > /dev/null && echo "✅ Ollama service is running" || echo "❌ Ollama service not running, please start: ollama serve"

# Interactive test
interactive:
	$(VENV_ACTIVATE) python codebase_agent_rag.py chat

# Debug specific query (requires providing query parameter)
debug:
	@if [ -z "$(QUERY)" ]; then \
		echo "Please provide query parameter, e.g.: make debug QUERY='How to implement this feature?'"; \
	else \
		$(VENV_ACTIVATE) python codebase_agent_rag.py search "$(QUERY)"; \
	fi

# Quick test (run basic tests only)
quick-test:
	@echo "Running quick tests..."
	$(VENV_ACTIVATE) python codebase_agent_rag.py test-rag-decision "Quick test query"
	make syntax
	make imports

# Development mode (clean + test)
dev:
	make clean
	make test

# Installation and setup
setup:
	make deps
	make ollama
	make syntax
	make imports 