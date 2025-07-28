#!/bin/bash

# Codebase Agent RAG 安装脚本

echo "🚀 开始安装 Codebase Agent RAG..."

# 检查 Python 版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ 错误: 需要 Python 3.8 或更高版本"
    echo "当前版本: $python_version"
    exit 1
fi

echo "✅ Python 版本检查通过: $python_version"

# 创建虚拟环境
echo "📦 创建虚拟环境..."
python3 -m venv venv_rag

# 激活虚拟环境
echo "🔄 激活虚拟环境..."
source venv_rag/bin/activate

# 升级 pip
echo "📈 升级 pip..."
pip install --upgrade pip

# 安装依赖
echo "📚 安装依赖（这可能需要几分钟）..."
pip install -r requirements_rag.txt

# 下载嵌入模型
echo "🤖 下载嵌入模型..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 使脚本可执行
chmod +x codebase_agent_rag.py

# 创建别名脚本
echo "🔧 创建命令行工具..."
cat > codebase-agent-rag << 'EOF'
#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/venv_rag/bin/activate"
python "$SCRIPT_DIR/codebase_agent_rag.py" "$@"
EOF

chmod +x codebase-agent-rag

echo ""
echo "✅ 安装完成！"
echo ""
echo "📝 使用说明："
echo "1. 确保 Ollama 正在运行:"
echo "   ollama serve"
echo ""
echo "2. 安装推荐的模型:"
echo "   ollama pull deepseek-coder:6.7b"
echo ""
echo "3. 索引代码库:"
echo "   ./codebase-agent-rag index"
echo ""
echo "4. 使用 RAG 搜索和分析:"
echo "   ./codebase-agent-rag search '数据库连接'"
echo "   ./codebase-agent-rag explain '这个项目如何处理用户认证？'"
echo "   ./codebase-agent-rag chat"
echo ""
echo "查看所有命令:"
echo "   ./codebase-agent-rag --help"