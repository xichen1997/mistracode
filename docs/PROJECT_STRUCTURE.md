# Codebase Agent RAG Project Structure

## Directory Structure

```
mistralcode/
├── codebase_agent_rag.py      # Main program file
├── build.py                   # Python build system
├── Makefile                   # Makefile shortcuts
├── requirements.txt           # Python dependencies
├── README_RAG_IMPROVEMENTS.md # RAG improvements documentation
├── PROJECT_STRUCTURE.md       # Project structure documentation
└── tests/                     # Test directory
    ├── __init__.py           # Test package initialization
    ├── conftest.py           # pytest configuration
    ├── run_tests.py          # Test runner
    ├── test_rag_decision.py  # RAG decision tests
    ├── test_llm_judgment.py  # LLM judgment tests
    └── test_improved_rag.py  # Improved RAG tests
```

## Build System

### Python Build System (build.py)

Provides Makefile-like functionality, supporting the following commands:

- `python build.py build` - Complete build process
- `python build.py test [name]` - Run tests
- `python build.py syntax` - Syntax check
- `python build.py imports` - Import test
- `python build.py clean` - Clean temporary files
- `python build.py deps` - Install dependencies
- `python build.py ollama` - Check Ollama service
- `python build.py interactive` - Interactive test
- `python build.py debug <query>` - Debug specific query

### Makefile Shortcuts

Provides traditional Makefile interface:

- `make build` - Complete build process
- `make test` - Run all tests
- `make test-rag` - Run RAG decision tests
- `make test-llm` - Run LLM judgment tests
- `make test-improved` - Run improved RAG tests
- `make clean` - Clean temporary files
- `make deps` - Install dependencies
- `make ollama` - Check Ollama service
- `make interactive` - Interactive test
- `make debug QUERY="query"` - Debug specific query

## Test System

### Test Files

1. **test_rag_decision.py** - Basic RAG decision tests
   - Test keyword matching functionality
   - Verify query classification accuracy

2. **test_llm_judgment.py** - LLM judgment functionality tests
   - Test LLM intelligent judgment
   - Includes interactive test mode

3. **test_improved_rag.py** - Improved RAG functionality tests
   - Comprehensive testing of improved features
   - Includes edge case testing

### Test Runner

- **tests/run_tests.py** - Independent test runner
- **tests/conftest.py** - pytest configuration file
- Supports unit test and integration test markers

## Usage

### Quick Start

```bash
# Complete build and test
make build

# Or use Python build system
python build.py build
```

### Development Workflow

```bash
# 1. Clean environment
make clean

# 2. Install dependencies
make deps

# 3. Check service
make ollama

# 4. Run tests
make test

# 5. Debug specific issues
make debug QUERY="How to implement this feature?"
```

### Test Specific Features

```bash
# Test RAG decision
make test-rag

# Test LLM judgment
make test-llm

# Test improved features
make test-improved

# Interactive test
make interactive
```

## Advantages

1. **Unified Interface** - Provides both Python and Makefile interfaces
2. **Complete Process** - Complete workflow from dependency installation to test execution
3. **Detailed Feedback** - Rich output information and error reporting
4. **Flexible Configuration** - Supports multiple test modes and configurations
5. **Easy to Extend** - Modular design, easy to add new features

## Extension

### Adding New Tests

1. Create new test files in the `tests/` directory
2. Add new tests to the `test_files` list in `build.py`
3. Add corresponding targets in `Makefile`

### Adding New Commands

1. Add new methods to the `BuildSystem` class
2. Add command mapping to the `commands` dictionary
3. Add corresponding targets in `Makefile` 