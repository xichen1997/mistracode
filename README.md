# Codebase Agent RAG

基于向量检索增强生成（RAG）的智能代码库助手，使用 Ollama 本地运行，提供更准确的代码理解和分析能力。

## 🚀 新功能特性

### RAG 增强功能
- **向量化索引**: 将代码库转换为向量存储，支持语义搜索
- **智能代码分块**: 自动识别函数、类等代码结构
- **增量索引**: 只更新修改过的文件，提高效率
- **上下文增强**: 修改代码时自动查找相关代码作为参考
- **持久化存储**: 索引数据本地保存，无需重复构建

### 核心功能
- 🔍 **语义搜索**: 基于含义而非关键词搜索代码
- 📊 **精准定位**: 快速找到相关函数、类和代码片段
- 💡 **智能解释**: 结合相关代码上下文提供准确解释
- ✏️ **上下文修改**: 修改代码时参考相关实现
- 💬 **交互式分析**: 支持连续对话和深度分析

## 安装

### 1. 系统要求
- Python 3.8+
- 4GB+ RAM（推荐 8GB）
- 2GB 磁盘空间（用于模型和索引）

### 2. 安装 Ollama
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

### 3. 启动 Ollama 并下载模型
```bash
# 启动服务
ollama serve

# 下载推荐模型（新终端）
ollama pull deepseek-coder:6.7b
```

### 4. 安装 Codebase Agent RAG
```bash
# 运行安装脚本
chmod +x setup_rag.sh
./setup_rag.sh
```

## 快速开始

### 1. 首次使用 - 建立索引
```bash
# 索引当前目录
./codebase-agent-rag index

# 索引指定目录
./codebase-agent-rag index --path /path/to/project

# 强制重建索引
./codebase-agent-rag index --force
```

### 2. 搜索代码
```bash
# 语义搜索
./codebase-agent-rag search "数据库连接"
./codebase-agent-rag search "用户认证逻辑"
./codebase-agent-rag search "API 错误处理"

# 指定结果数量
./codebase-agent-rag search "缓存实现" -n 20
```

### 3. 解释代码（RAG 增强）
```bash
# 解释功能
./codebase-agent-rag explain "这个项目是如何实现用户认证的？"
./codebase-agent-rag explain "数据库事务是如何处理的？"

# 使用更多搜索结果
./codebase-agent-rag explain "性能优化策略" -n 15
```

### 4. 修改代码（上下文增强）
```bash
# 修改文件
./codebase-agent-rag modify app.py "添加请求限流功能"

# 预览模式
./codebase-agent-rag modify config.py "添加 Redis 配置" --dry-run
```

### 5. 交互式分析
```bash
# 进入交互模式
./codebase-agent-rag chat

# 交互模式命令:
# /stats - 显示索引统计
# /help - 显示帮助
# /clear - 清屏
```

## 工作原理

### 1. 代码解析
- 使用 AST 解析 Python 代码，识别函数和类
- 智能分块，保持代码结构完整性
- 提取文档字符串和元数据

### 2. 向量化索引
- 使用 Sentence Transformers 生成代码嵌入
- ChromaDB 存储向量和元数据
- 余弦相似度计算相关性

### 3. RAG 流程
```
用户查询 -> 向量化 -> 搜索相似代码 -> 构建上下文 -> LLM 生成答案
```

## 高级功能

### 查看索引统计
```bash
./codebase-agent-rag stats
```

### 列出已索引文件
```bash
./codebase-agent-rag list-files
```

### 清除索引
```bash
./codebase-agent-rag clear
```

### 使用不同模型
```bash
# 使用 Mistral
./codebase-agent-rag --model mistral explain "代码架构"

# 使用 CodeLlama
./codebase-agent-rag --model codellama search "设计模式"
```

## 性能优化

### 1. 索引优化
- **增量更新**: 只索引修改的文件
- **并行处理**: 自动使用多核处理
- **智能分块**: 根据代码结构优化块大小

### 2. 搜索优化
- **向量缓存**: 减少重复计算
- **预过滤**: 基于文件类型和路径优化搜索
- **相关度阈值**: 只返回高相关度结果

### 3. 内存优化
- **流式处理**: 大文件分批处理
- **按需加载**: 只加载必要的代码片段
- **垃圾回收**: 自动清理未使用的数据

## 常见问题

**Q: 索引需要多长时间？**
A: 取决于代码库大小。1000 个文件约需 2-5 分钟。

**Q: 索引占用多少空间？**
A: 通常是代码库大小的 20-50%。

**Q: 如何更新索引？**
A: 再次运行 `index` 命令会自动增量更新。

**Q: 支持哪些编程语言？**
A: 所有文本格式的编程语言。Python 有最佳支持。

**Q: 可以索引多个项目吗？**
A: 可以，使用不同的 `--index-dir` 参数。

## 与基础版本对比

| 功能 | 基础版本 | RAG 版本 |
|------|---------|----------|
| 代码搜索 | 关键词匹配 | 语义搜索 |
| 上下文大小 | 受限于 Token | 智能选择相关片段 |
| 大型代码库 | 可能遗漏文件 | 完整索引 |
| 搜索速度 | 每次扫描 | 毫秒级响应 |
| 准确性 | 一般 | 高 |

## 故障排除

### Ollama 连接失败
```bash
# 检查 Ollama 状态
curl http://localhost:11434/api/tags

# 重启 Ollama
ollama serve
```

### 内存不足
- 减少批处理大小
- 使用更小的嵌入模型
- 清理未使用的索引

### 索引错误
```bash
# 清除损坏的索引
./codebase-agent-rag clear -y

# 重新索引
./codebase-agent-rag index --force
```
## DEBUG 工具

● 现在您可以使用这些调试工具来查看索引的详细信息：

  1. 查看所有检查（推荐先用这个）：
  python debug_utilities.py all

  2. 检查集合基本信息：
  python debug_utilities.py inspect

  3. 查看样本文档：
  python debug_utilities.py samples --limit 10

  4. 分析文件分布：
  python debug_utilities.py distribution

  5. 调试搜索功能：
  python debug_utilities.py search-debug "your query"

  6. 查看文件哈希缓存：
  python debug_utilities.py hashes

  7. 验证索引完整性：
  python debug_utilities.py validate

  8. 导出索引数据到 JSON：
  python debug_utilities.py export --output my_index.json

  这些工具可以帮助您：
  - 查看索引中有多少文档和文件
  - 检查元数据字段是否正确
  - 验证搜索功能是否正常工作
  - 发现索引中的问题
  - 导出数据进行进一步分析

  试试运行 python debug_utilities.py all 来获得完整的索引概览！


## 开发路线图

- [ ] 支持更多编程语言的 AST 解析
- [ ] 代码依赖关系图
- [ ] Git 历史集成
- [ ] 多项目联合搜索
- [ ] Web UI 界面
- [ ] 团队协作功能

## 贡献

欢迎提交 Issue 和 Pull Request！
