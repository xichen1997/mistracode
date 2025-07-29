#!/bin/bash

# Codebase Agent RAG installation script

echo "🚀 Starting installation of Codebase Agent RAG..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ Error: Python 3.8 or higher required"
    echo "Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv_rag

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv_rag/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies (this may take a few minutes)..."
pip install -r requirements_rag.txt

# Download embedding model
echo "🤖 Downloading embedding model..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Make script executable
chmod +x codebase_agent_rag.py

# Create alias script
echo "🔧 Creating command line tool..."
cat > codebase-agent-rag << 'EOF'
#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/venv_rag/bin/activate"
python "$SCRIPT_DIR/codebase_agent_rag.py" "$@"
EOF

chmod +x codebase-agent-rag

echo ""
echo "✅ Installation complete!"
echo ""
echo "📝 Usage instructions:"
echo "1. Ensure Ollama is running:"
echo "   ollama serve"
echo ""
echo "2. Install recommended model:"
echo "   ollama pull deepseek-coder:6.7b"
echo ""
echo "3. Index codebase:"
echo "   ./codebase-agent-rag index"
echo ""
echo "4. Use RAG search and analysis:"
echo "   ./codebase-agent-rag search 'database connection'"
echo "   ./codebase-agent-rag explain 'How does this project handle user authentication?'"
echo "   ./codebase-agent-rag chat"
echo ""
echo "View all commands:"
echo "   ./codebase-agent-rag --help"