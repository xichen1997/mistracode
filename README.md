# Codebase Agent RAG

An intelligent codebase assistant based on Retrieval-Augmented Generation (RAG), running locally with Ollama to provide more accurate code understanding and analysis capabilities.

## 📋 Table of Contents

- [🚀 New Features](#-new-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [🛠️ Makefile Build System](#️-makefile-build-system)
- [How It Works](#how-it-works)
- [Advanced Features](#advanced-features)
- [Performance Optimization](#performance-optimization)
- [FAQ](#faq)
- [Comparison with Base Version](#comparison-with-base-version)
- [Troubleshooting](#troubleshooting)
- [DEBUG Tools](#debug-tools)
- [Development Roadmap](#development-roadmap)
- [Contributing](#contributing)

## 🚀 New Features

### RAG Enhancement Features
- **Vector Indexing**: Convert codebase to vector storage for semantic search
- **Smart Code Chunking**: Automatically identify functions, classes and other code structures
- **Incremental Indexing**: Only update modified files for improved efficiency
- **Context Enhancement**: Automatically find related code as reference when modifying code
- **Persistent Storage**: Index data saved locally, no need to rebuild

### Core Features
- 🔍 **Semantic Search**: Search code based on meaning rather than keywords
- 📊 **Precise Location**: Quickly find relevant functions, classes and code snippets
- 💡 **Smart Explanation**: Provide accurate explanations combined with relevant code context
- ✏️ **Context Modification**: Reference related implementations when modifying code
- 💬 **Interactive Analysis**: Support continuous dialogue and in-depth analysis

## Installation

### 1. System Requirements
- Python 3.8+
- 4GB+ RAM (8GB recommended)
- 2GB disk space (for models and indices)

### 2. Install Ollama
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

### 3. Start Ollama and Download Models
```bash
# Start service
ollama serve

# Download recommended model (new terminal)
ollama pull deepseek-coder:6.7b
```

### 4. Install Codebase Agent RAG
```bash
# Run installation script
chmod +x setup_rag.sh
./setup_rag.sh
```

## Quick Start

### 1. First Use - Build Index
```bash
# Index current directory
./codebase-agent-rag index

# Index specific directory
./codebase-agent-rag index --path /path/to/project

# Force rebuild index
./codebase-agent-rag index --force
```

### 2. Search Code
```bash
# Semantic search
./codebase-agent-rag search "database connection"
./codebase-agent-rag search "user authentication logic"
./codebase-agent-rag search "API error handling"

# Specify number of results
./codebase-agent-rag search "cache implementation" -n 20
```

### 3. Explain Code (RAG Enhanced)
```bash
# Explain features
./codebase-agent-rag explain "How does this project implement user authentication?"
./codebase-agent-rag explain "How are database transactions handled?"

# Use more search results
./codebase-agent-rag explain "performance optimization strategies" -n 15
```

### 4. Modify Code (Context Enhanced)
```bash
# Modify file
./codebase-agent-rag modify app.py "Add request rate limiting feature"

# Preview mode
./codebase-agent-rag modify config.py "Add Redis configuration" --dry-run
```

### 5. Interactive Analysis
```bash
# Enter interactive mode
./codebase-agent-rag chat

# Interactive mode commands:
# /stats - Show index statistics
# /help - Show help
# /clear - Clear screen
```

## 🛠️ Makefile Build System

The project provides a convenient Makefile to simplify development and testing workflows. All commands will automatically activate the virtual environment.

### Basic Commands

```bash
# Show all available commands
make help

# Complete build process (build index)
make build

# Run all tests
make test

# Clean temporary files
make clean

# Install dependencies
make deps

# Check Ollama service status
make ollama
```

### Test Commands

```bash
# Run RAG decision tests
make test-rag

# Run LLM judgment tests
make test-llm

# Run improved RAG tests
make test-improved

# Quick test (basic test + syntax check + import test)
make quick-test
```

### Code Quality Checks

```bash
# Syntax check
make syntax

# Import test
make imports
```

### Development and Debugging

```bash
# Interactive test (chat mode)
make interactive

# Debug specific query
make debug QUERY="How to implement user authentication feature?"

# Development mode (clean + test)
make dev

# Complete setup (dependencies + Ollama + syntax + imports)
make setup
```

### Command Description

| Command | Function | Description |
|---------|----------|-------------|
| `make build` | Build codebase index | Run `codebase_agent_rag.py index` |
| `make test` | Run complete test suite | Include RAG decision and LLM judgment tests |
| `make test-rag` | RAG decision test | Test if RAG functionality works properly |
| `make test-llm` | LLM judgment test | Test LLM judgment functionality |
| `make syntax` | Syntax check | Check Python file syntax correctness |
| `make imports` | Import test | Verify all modules can be imported properly |
| `make clean` | Clean files | Delete cache and temporary files |
| `make deps` | Install dependencies | Install dependencies from requirements_rag.txt |
| `make ollama` | Check service | Verify if Ollama service is running |
| `make build` | Build index | Build codebase index using Ollama embedding model |
| `make interactive` | Interactive mode | Start chat mode for code analysis |
| `make debug` | Debug query | Requires QUERY parameter |

### Usage Examples

```bash
# First time project setup
make setup

# Daily development workflow
make clean
make test
make syntax

# Debug specific issues
make debug QUERY="What is the cause of database connection failure?"

# Quick validation
make quick-test
```

### Notes

- All commands automatically activate the `venv_rag` virtual environment
- Test commands use preset example queries, no manual input required
- `make debug` requires providing the `QUERY` parameter to run
- `make setup` executes the complete project initialization process

## How It Works

### 1. Code Parsing
- Use AST to parse Python code, identify functions and classes
- Smart chunking to maintain code structure integrity
- Extract docstrings and metadata

### 2. Vector Indexing
- Use Sentence Transformers to generate code embeddings
- ChromaDB stores vectors and metadata
- Cosine similarity calculates relevance

### 3. RAG Process
```
User Query -> Vectorize -> Search Similar Code -> Build Context -> LLM Generate Answer
```

## Advanced Features

### View Index Statistics
```bash
./codebase-agent-rag stats
```

### List Indexed Files
```bash
./codebase-agent-rag list-files
```

### Clear Index
```bash
./codebase-agent-rag clear
```

### Use Different Models
```bash
# Use Mistral
./codebase-agent-rag --model mistral explain "code architecture"

# Use CodeLlama
./codebase-agent-rag --model codellama search "design patterns"

# Use Devstral (supports embedding)
./codebase-agent-rag --model devstral search "user authentication"
```

### Embedding Model Support

The project now supports using Ollama models for text embedding, providing better semantic search capabilities:

- **Devstral**: Supports 5120-dimensional embedding vectors, providing high-quality semantic understanding
- **Nomic Embed**: Specialized embedding model optimized for text similarity computation
- **Automatic Fallback**: If Ollama model doesn't support embedding, automatically falls back to SentenceTransformer

Advantages of using Ollama embedding:
- **Consistency**: Use the same model for generation and embedding, ensuring semantic consistency
- **Local**: Runs completely locally, protecting code privacy
- **Performance**: Embedding vectors optimized for code understanding

## Performance Optimization

### 1. Index Optimization
- **Incremental Updates**: Only index modified files
- **Parallel Processing**: Automatically use multi-core processing
- **Smart Chunking**: Optimize chunk size based on code structure

### 2. Search Optimization
- **Vector Caching**: Reduce duplicate calculations
- **Pre-filtering**: Optimize search based on file types and paths
- **Relevance Threshold**: Only return high relevance results

### 3. Memory Optimization
- **Stream Processing**: Process large files in batches
- **On-demand Loading**: Only load necessary code snippets
- **Garbage Collection**: Automatically clean unused data

## FAQ

**Q: How long does indexing take?**
A: Depends on codebase size. About 2-5 minutes for 1000 files.

**Q: How much space does the index occupy?**
A: Usually 20-50% of the codebase size.

**Q: How to update the index?**
A: Running the `index` command again will automatically perform incremental updates.

**Q: What programming languages are supported?**
A: All text-format programming languages. Python has the best support.

**Q: Can multiple projects be indexed?**
A: Yes, use different `--index-dir` parameters.

## Comparison with Base Version

| Feature | Base Version | RAG Version |
|---------|-------------|-------------|
| Code Search | Keyword matching | Semantic search |
| Context Size | Limited by Token | Smart selection of relevant snippets |
| Large Codebase | May miss files | Complete indexing |
| Search Speed | Scan each time | Millisecond response |
| Accuracy | General | High |

## Troubleshooting

### Ollama Connection Failed
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### Memory Insufficient
- Reduce batch processing size
- Use smaller embedding models
- Clean unused indices

### Index Errors
```bash
# Clear corrupted index
./codebase-agent-rag clear -y

# Re-index
./codebase-agent-rag index --force
```
## DEBUG Tools

● You can now use these debugging tools to view detailed index information:

  1. View all checks (recommended to use this first):
  python debug_utilities.py all

  2. Check collection basic information:
  python debug_utilities.py inspect

  3. View sample documents:
  python debug_utilities.py samples --limit 10

  4. Analyze file distribution:
  python debug_utilities.py distribution

  5. Debug search functionality:
  python debug_utilities.py search-debug "your query"

  6. View file hash cache:
  python debug_utilities.py hashes

  7. Validate index integrity:
  python debug_utilities.py validate

  8. Export index data to JSON:
  python debug_utilities.py export --output my_index.json

  These tools can help you:
  - View how many documents and files are in the index
  - Check if metadata fields are correct
  - Verify if search functionality works properly
  - Discover issues in the index
  - Export data for further analysis

  Try running python debug_utilities.py all to get a complete index overview!


## Development Roadmap

- [ ] Support AST parsing for more programming languages
- [ ] Code dependency relationship graph
- [ ] Git history integration
- [ ] Multi-project joint search
- [ ] Web UI interface
- [ ] Team collaboration features

## Contributing

Welcome to submit Issues and Pull Requests!
