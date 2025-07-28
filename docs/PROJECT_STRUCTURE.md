# Codebase Agent RAG 项目结构

## 目录结构

```
mistralcode/
├── codebase_agent_rag.py      # 主程序文件
├── build.py                   # Python 构建系统
├── Makefile                   # Makefile 快捷方式
├── requirements.txt           # Python 依赖
├── README_RAG_IMPROVEMENTS.md # RAG 改进说明
├── PROJECT_STRUCTURE.md       # 项目结构说明
└── tests/                     # 测试目录
    ├── __init__.py           # 测试包初始化
    ├── conftest.py           # pytest 配置
    ├── run_tests.py          # 测试运行器
    ├── test_rag_decision.py  # RAG 决策测试
    ├── test_llm_judgment.py  # LLM 判断测试
    └── test_improved_rag.py  # 改进后的 RAG 测试
```

## 构建系统

### Python 构建系统 (build.py)

提供类似 Makefile 的功能，支持以下命令：

- `python build.py build` - 完整构建流程
- `python build.py test [name]` - 运行测试
- `python build.py syntax` - 语法检查
- `python build.py imports` - 导入测试
- `python build.py clean` - 清理临时文件
- `python build.py deps` - 安装依赖
- `python build.py ollama` - 检查 Ollama 服务
- `python build.py interactive` - 交互式测试
- `python build.py debug <query>` - 调试特定查询

### Makefile 快捷方式

提供传统的 Makefile 接口：

- `make build` - 完整构建流程
- `make test` - 运行所有测试
- `make test-rag` - 运行 RAG 决策测试
- `make test-llm` - 运行 LLM 判断测试
- `make test-improved` - 运行改进后的 RAG 测试
- `make clean` - 清理临时文件
- `make deps` - 安装依赖
- `make ollama` - 检查 Ollama 服务
- `make interactive` - 交互式测试
- `make debug QUERY="查询"` - 调试特定查询

## 测试系统

### 测试文件

1. **test_rag_decision.py** - 基本 RAG 决策测试
   - 测试关键词匹配功能
   - 验证查询分类准确性

2. **test_llm_judgment.py** - LLM 判断功能测试
   - 测试 LLM 智能判断
   - 包含交互式测试模式

3. **test_improved_rag.py** - 改进后的 RAG 功能测试
   - 全面测试改进后的功能
   - 包含边界情况测试

### 测试运行器

- **tests/run_tests.py** - 独立的测试运行器
- **tests/conftest.py** - pytest 配置文件
- 支持单元测试和集成测试标记

## 使用方法

### 快速开始

```bash
# 完整构建和测试
make build

# 或者使用 Python 构建系统
python build.py build
```

### 开发工作流

```bash
# 1. 清理环境
make clean

# 2. 安装依赖
make deps

# 3. 检查服务
make ollama

# 4. 运行测试
make test

# 5. 调试特定问题
make debug QUERY="如何实现这个功能？"
```

### 测试特定功能

```bash
# 测试 RAG 决策
make test-rag

# 测试 LLM 判断
make test-llm

# 测试改进后的功能
make test-improved

# 交互式测试
make interactive
```

## 优势

1. **统一接口** - 提供 Python 和 Makefile 两种接口
2. **完整流程** - 从依赖安装到测试运行的完整流程
3. **详细反馈** - 丰富的输出信息和错误报告
4. **灵活配置** - 支持多种测试模式和配置
5. **易于扩展** - 模块化设计，易于添加新功能

## 扩展

### 添加新测试

1. 在 `tests/` 目录下创建新的测试文件
2. 在 `build.py` 的 `test_files` 列表中添加新测试
3. 在 `Makefile` 中添加对应的目标

### 添加新命令

1. 在 `BuildSystem` 类中添加新方法
2. 在 `commands` 字典中添加命令映射
3. 在 `Makefile` 中添加对应的目标 