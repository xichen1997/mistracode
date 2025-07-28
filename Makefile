# Codebase Agent RAG Makefile
# 基于 Python 构建系统的快捷方式

.PHONY: help build test clean deps syntax imports ollama debug interactive

# 虚拟环境激活
VENV_ACTIVATE = . venv_rag/bin/activate &&

# 默认目标
help:
	@echo "Codebase Agent RAG 构建系统"
	@echo ""
	@echo "可用命令:"
	@echo "  make build        - 完整构建流程"
	@echo "  make test         - 运行所有测试"
	@echo "  make test-rag     - 运行 RAG 决策测试"
	@echo "  make test-llm     - 运行 LLM 判断测试"
	@echo "  make test-improved - 运行改进后的 RAG 测试"
	@echo "  make syntax       - 语法检查"
	@echo "  make imports      - 导入测试"
	@echo "  make clean        - 清理临时文件"
	@echo "  make deps         - 安装依赖"
	@echo "  make ollama       - 检查 Ollama 服务"
	@echo "  make interactive  - 交互式测试"
	@echo "  make debug        - 调试特定查询"
	@echo "  make help         - 显示帮助"

# 完整构建流程
build:
	$(VENV_ACTIVATE) python codebase_agent_rag.py index

# 运行所有测试
test:
	@echo "运行 RAG 决策测试..."
	$(VENV_ACTIVATE) python codebase_agent_rag.py test-rag-decision "如何实现用户认证功能？"
	@echo "运行 LLM 判断测试..."
	$(VENV_ACTIVATE) python codebase_agent_rag.py debug-llm-judgment "这个项目的主要功能是什么？"

# 运行特定测试
test-rag:
	@echo "运行 RAG 决策测试..."
	$(VENV_ACTIVATE) python codebase_agent_rag.py test-rag-decision "如何实现用户认证功能？"

test-llm:
	@echo "运行 LLM 判断测试..."
	$(VENV_ACTIVATE) python codebase_agent_rag.py debug-llm-judgment "这个项目的主要功能是什么？"

test-improved:
	@echo "运行改进后的 RAG 测试..."
	$(VENV_ACTIVATE) python codebase_agent_rag.py test-rag-decision "如何优化代码性能？"

# 语法检查
syntax:
	$(VENV_ACTIVATE) python -m py_compile codebase_agent_rag.py
	$(VENV_ACTIVATE) python -m py_compile debug_utilities.py

# 导入测试
imports:
	$(VENV_ACTIVATE) python -c "import codebase_agent_rag; import debug_utilities; print('✅ 所有导入成功')"

# 清理临时文件
clean:
	rm -rf __pycache__
	rm -rf .codebase_index
	rm -rf *.pyc
	rm -rf .pytest_cache
	rm -rf .coverage

# 安装依赖
deps:
	$(VENV_ACTIVATE) pip install -r requirements_rag.txt

# 检查 Ollama 服务
ollama:
	@echo "检查 Ollama 服务状态..."
	@curl -s http://localhost:11434/api/tags > /dev/null && echo "✅ Ollama 服务正在运行" || echo "❌ Ollama 服务未运行，请启动: ollama serve"

# 交互式测试
interactive:
	$(VENV_ACTIVATE) python codebase_agent_rag.py chat

# 调试特定查询 (需要提供查询参数)
debug:
	@if [ -z "$(QUERY)" ]; then \
		echo "请提供查询参数，例如: make debug QUERY='如何实现这个功能？'"; \
	else \
		$(VENV_ACTIVATE) python codebase_agent_rag.py search "$(QUERY)"; \
	fi

# 快速测试 (只运行基本测试)
quick-test:
	@echo "运行快速测试..."
	$(VENV_ACTIVATE) python codebase_agent_rag.py test-rag-decision "快速测试查询"
	make syntax
	make imports

# 开发模式 (清理 + 测试)
dev:
	make clean
	make test

# 安装和设置
setup:
	make deps
	make ollama
	make syntax
	make imports 