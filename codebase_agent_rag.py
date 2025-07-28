#!/usr/bin/env python3
"""
Codebase Agent with RAG - 使用 Ollama 和向量数据库的代码解释和修改工具
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import click
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import ast

console = Console()

class OllamaClient:
    """Ollama API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "devstral:latest"):
        self.base_url = base_url
        self.model = model
        
    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 4000) -> str:
        """调用 Ollama 生成响应"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens
                    }
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.ConnectionError:
            raise Exception("无法连接到 Ollama 服务。请确保 Ollama 正在运行。")
        except Exception as e:
            raise Exception(f"Ollama API 调用失败: {str(e)}")
    
    def embeddings(self, text: str) -> List[float]:
        """获取文本的嵌入向量"""
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except:
            # 如果 Ollama 不支持嵌入，返回 None
            return None
    
    def list_models(self) -> List[str]:
        """列出可用的模型"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except:
            return []

class CodeChunk:
    """代码片段类"""
    
    def __init__(self, content: str, file_path: str, start_line: int, end_line: int, 
                 chunk_type: str = "code", metadata: Dict = None):
        self.content = content
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.chunk_type = chunk_type  # function, class, module, code
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type,
            "metadata": self.metadata
        }

class CodeParser:
    """代码解析器，将代码文件分解为有意义的片段"""
    
    @staticmethod
    def parse_python(content: str, file_path: str) -> List[CodeChunk]:
        """解析 Python 代码"""
        chunks = []
        
        try:
            tree = ast.parse(content)
            
            # 提取函数
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    start_line = node.lineno
                    end_line = node.end_lineno or start_line
                    
                    # 获取函数代码
                    lines = content.splitlines()
                    func_lines = lines[start_line-1:end_line]
                    func_content = '\n'.join(func_lines)
                    
                    # 提取文档字符串
                    docstring = ast.get_docstring(node) or ""
                    
                    chunk = CodeChunk(
                        content=func_content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type="function",
                        metadata={
                            "name": node.name,
                            "docstring": docstring,
                            "args": ", ".join([arg.arg for arg in node.args.args])
                        }
                    )
                    chunks.append(chunk)
                
                elif isinstance(node, ast.ClassDef):
                    start_line = node.lineno
                    end_line = node.end_lineno or start_line
                    
                    # 获取类代码
                    lines = content.splitlines()
                    class_lines = lines[start_line-1:end_line]
                    class_content = '\n'.join(class_lines)
                    
                    # 提取文档字符串
                    docstring = ast.get_docstring(node) or ""
                    
                    chunk = CodeChunk(
                        content=class_content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_type="class",
                        metadata={
                            "name": node.name,
                            "docstring": docstring,
                            "methods": ", ".join([m.name for m in node.body if isinstance(m, ast.FunctionDef)])
                        }
                    )
                    chunks.append(chunk)
        except:
            # 如果解析失败，返回整个文件作为一个块
            pass
        
        # 如果没有找到函数或类，或者解析失败，将整个文件分成较小的块
        if not chunks:
            lines = content.splitlines()
            chunk_size = 50  # 每个块的行数
            
            for i in range(0, len(lines), chunk_size):
                chunk_lines = lines[i:i+chunk_size]
                chunk = CodeChunk(
                    content='\n'.join(chunk_lines),
                    file_path=file_path,
                    start_line=i+1,
                    end_line=min(i+chunk_size, len(lines)),
                    chunk_type="code"
                )
                chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def parse_javascript(content: str, file_path: str) -> List[CodeChunk]:
        """解析 JavaScript/TypeScript 代码（简化版）"""
        chunks = []
        lines = content.splitlines()
        
        # 简单的函数和类检测
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # 检测函数
            if ('function ' in line or 'const ' in line or 'let ' in line or 'var ' in line) and ('=' in line or '(' in line):
                start = i
                brace_count = 0
                found_brace = False
                
                # 找到函数结束
                for j in range(i, len(lines)):
                    if '{' in lines[j]:
                        brace_count += lines[j].count('{')
                        found_brace = True
                    if '}' in lines[j]:
                        brace_count -= lines[j].count('}')
                    
                    if found_brace and brace_count == 0:
                        chunk = CodeChunk(
                            content='\n'.join(lines[start:j+1]),
                            file_path=file_path,
                            start_line=start+1,
                            end_line=j+1,
                            chunk_type="function"
                        )
                        chunks.append(chunk)
                        i = j + 1
                        break
                else:
                    i += 1
            else:
                i += 1
        
        # 如果没有找到函数，分块处理
        if not chunks:
            chunk_size = 50
            for i in range(0, len(lines), chunk_size):
                chunk = CodeChunk(
                    content='\n'.join(lines[i:i+chunk_size]),
                    file_path=file_path,
                    start_line=i+1,
                    end_line=min(i+chunk_size, len(lines)),
                    chunk_type="code"
                )
                chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def parse_generic(content: str, file_path: str) -> List[CodeChunk]:
        """通用代码解析（按行数分块）"""
        chunks = []
        lines = content.splitlines()
        chunk_size = 50  # 每个块的行数
        
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i+chunk_size]
            chunk = CodeChunk(
                content='\n'.join(chunk_lines),
                file_path=file_path,
                start_line=i+1,
                end_line=min(i+chunk_size, len(lines)),
                chunk_type="code"
            )
            chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def parse_file(content: str, file_path: str) -> List[CodeChunk]:
        """根据文件类型解析代码"""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.py':
            return CodeParser.parse_python(content, file_path)
        elif ext in ['.js', '.jsx', '.ts', '.tsx']:
            return CodeParser.parse_javascript(content, file_path)
        else:
            return CodeParser.parse_generic(content, file_path)

class CodebaseRAG:
    """基于 RAG 的代码库管理系统"""
    
    def __init__(self, persist_dir: str = ".codebase_index"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)
        
        # 初始化嵌入模型
        console.print("[cyan]正在加载嵌入模型...[/cyan]")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 初始化向量数据库
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir / "chroma"),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 创建或加载集合
        try:
            self.collection = self.chroma_client.get_collection("codebase")
            console.print("[green]加载已有索引[/green]")
        except:
            self.collection = self.chroma_client.create_collection(
                name="codebase",
                metadata={"hnsw:space": "cosine"}
            )
            console.print("[yellow]创建新索引[/yellow]")
        
        # 文件哈希缓存（用于检测文件变化）
        self.hash_cache_file = self.persist_dir / "file_hashes.json"
        self.file_hashes = self._load_file_hashes()
    
    def _load_file_hashes(self) -> Dict[str, str]:
        """加载文件哈希缓存"""
        if self.hash_cache_file.exists():
            with open(self.hash_cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_file_hashes(self):
        """保存文件哈希缓存"""
        with open(self.hash_cache_file, 'w') as f:
            json.dump(self.file_hashes, f)
    
    def _get_file_hash(self, file_path: str) -> str:
        """计算文件哈希"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _embed_text(self, text: str) -> List[float]:
        """生成文本嵌入"""
        return self.embedder.encode(text).tolist()
    
    def index_file(self, file_path: str, content: str):
        """索引单个文件"""
        # 检查文件是否已更改
        current_hash = self._get_file_hash(file_path)
        
        if file_path in self.file_hashes and self.file_hashes[file_path] == current_hash:
            return  # 文件未更改，跳过
        
        # 删除旧的索引
        try:
            results = self.collection.get(where={"file_path": file_path})
            if results['ids']:
                self.collection.delete(ids=results['ids'])
        except:
            pass
        
        # 解析代码为块
        chunks = CodeParser.parse_file(content, file_path)
        
        # 索引每个块
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_path}:{chunk.start_line}:{chunk.end_line}"
            
            # 准备索引内容
            index_content = f"File: {file_path}\n"
            if chunk.metadata.get('name'):
                index_content += f"Name: {chunk.metadata['name']}\n"
            if chunk.metadata.get('docstring'):
                index_content += f"Description: {chunk.metadata['docstring']}\n"
            index_content += f"\n{chunk.content}"
            
            # 生成嵌入
            embedding = self._embed_text(index_content)
            
            # 存储到向量数据库
            self.collection.add(
                embeddings=[embedding],
                documents=[chunk.content],
                metadatas=[{
                    "file_path": file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_type": chunk.chunk_type,
                    **chunk.metadata
                }],
                ids=[chunk_id]
            )
        
        # 更新文件哈希
        self.file_hashes[file_path] = current_hash
    
    def search(self, query: str, n_results: int = 10) -> List[Dict]:
        """搜索相关代码片段"""
        # 生成查询嵌入
        query_embedding = self._embed_text(query)
        
        # 搜索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # 格式化结果
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else 0
            })
        
        return formatted_results
    
    def get_stats(self) -> Dict:
        """获取索引统计信息"""
        count = self.collection.count()
        
        # 获取所有文档的元数据来统计
        all_results = self.collection.get(limit=count)
        
        file_set = set()
        chunk_types = {}
        
        for metadata in all_results['metadatas']:
            file_set.add(metadata['file_path'])
            chunk_type = metadata.get('chunk_type', 'unknown')
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        return {
            'total_chunks': count,
            'total_files': len(file_set),
            'chunk_types': chunk_types,
            'indexed_files': list(file_set)
        }
    
    def clear_index(self):
        """清除所有索引"""
        self.chroma_client.delete_collection("codebase")
        self.collection = self.chroma_client.create_collection(
            name="codebase",
            metadata={"hnsw:space": "cosine"}
        )
        self.file_hashes = {}
        self._save_file_hashes()

class CodebaseAgentRAG:
    """支持 RAG 的代码库智能助手"""
    
    def __init__(self, model: str = "devstral:latest", 
                 base_url: str = "http://localhost:11434",
                 index_dir: str = ".codebase_index"):
        self.client = OllamaClient(base_url=base_url, model=model)
        self.model = model
        self.rag = CodebaseRAG(persist_dir=index_dir)
        
        # 支持的文件扩展名
        self.supported_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', 
            '.h', '.hpp', '.cs', '.go', '.rs', '.php', '.rb', '.swift',
            '.kt', '.scala', '.r', '.m', '.mm', '.vue', '.dart', '.lua',
            '.sh', '.bash', '.zsh', '.fish', '.ps1', '.yaml', '.yml',
            '.json', '.xml', '.toml', '.ini', '.cfg', '.conf', '.md',
            '.txt', '.sql', '.dockerfile', '.makefile'
        }
        
        # 忽略的目录
        self.ignore_dirs = {
            '.git', '__pycache__', 'node_modules', '.venv', 'venv', 'venv_rag',
            'env', 'env_rag', 'virtualenv', 'conda', 'miniconda', 'anaconda',
            'dist', 'build', '.idea', '.vscode', 'target',
            'bin', 'obj', '.pytest_cache', '.mypy_cache', '.tox',
            'coverage', '.coverage', 'htmlcov', '.sass-cache',
            '.codebase_index',  # 忽略索引目录
            'site-packages', 'lib64', 'include', 'share',  # 虚拟环境常见目录
            '.DS_Store', 'Thumbs.db'  # 系统文件
        }
        
        # 检查 Ollama 连接
        self._check_ollama_connection()
    
    def _check_ollama_connection(self):
        """检查 Ollama 服务是否可用"""
        try:
            models = self.client.list_models()
            if not models:
                console.print("[yellow]警告: Ollama 中没有安装任何模型[/yellow]")
                console.print("[cyan]请运行: ollama pull devstral:latest[/cyan]")
            elif self.model not in models:
                console.print(f"[yellow]警告: 模型 {self.model} 未安装[/yellow]")
                console.print(f"[cyan]可用模型: {', '.join(models)}[/cyan]")
                console.print(f"[cyan]安装模型: ollama pull {self.model}[/cyan]")
        except Exception as e:
            console.print(f"[red]错误: {str(e)}[/red]")
            console.print("[cyan]请确保 Ollama 正在运行: ollama serve[/cyan]")
            exit(1)
    
    def scan_directory(self, path: Path, show_progress: bool = True) -> List[Dict]:
        """扫描目录获取所有代码文件"""
        files = []
        
        # 收集所有需要处理的文件
        all_files = []
        for file_path in path.rglob("*"):
            if (file_path.is_file() and 
                file_path.suffix.lower() in self.supported_extensions and
                not any(ignored in file_path.parts for ignored in self.ignore_dirs)):
                all_files.append(file_path)
        
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]扫描代码文件...", total=len(all_files))
                
                for file_path in all_files:
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        relative_path = file_path.relative_to(path)
                        
                        files.append({
                            'path': str(relative_path),
                            'full_path': str(file_path),
                            'content': content,
                            'size': len(content),
                            'lines': len(content.splitlines())
                        })
                        
                        progress.update(task, advance=1)
                    except Exception as e:
                        console.print(f"[yellow]警告: 无法读取文件 {file_path}: {e}[/yellow]")
        else:
            for file_path in all_files:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    relative_path = file_path.relative_to(path)
                    
                    files.append({
                        'path': str(relative_path),
                        'full_path': str(file_path),
                        'content': content,
                        'size': len(content),
                        'lines': len(content.splitlines())
                    })
                except Exception:
                    pass
        
        return files
    
    def index_codebase(self, path: Path):
        """索引整个代码库"""
        console.print(f"\n[bold cyan]正在索引代码库...[/bold cyan]")
        
        files = self.scan_directory(path)
        
        if not files:
            console.print("[red]未找到任何代码文件[/red]")
            return
        
        console.print(f"[green]找到 {len(files)} 个代码文件[/green]")
        
        # 索引文件
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]建立索引...", total=len(files))
            
            for file in files:
                try:
                    self.rag.index_file(file['full_path'], file['content'])
                    progress.update(task, advance=1)
                except Exception as e:
                    console.print(f"[yellow]索引失败 {file['path']}: {e}[/yellow]")
        
        # 保存文件哈希
        self.rag._save_file_hashes()
        
        # 显示统计
        stats = self.rag.get_stats()
        console.print(f"\n[green]索引完成！[/green]")
        console.print(f"总文件数: {stats['total_files']}")
        console.print(f"总代码块: {stats['total_chunks']}")
        
        if stats['chunk_types']:
            console.print("\n代码块类型分布:")
            for chunk_type, count in stats['chunk_types'].items():
                console.print(f"  {chunk_type}: {count}")
    
    def should_use_rag(self, query: str) -> bool:
        """智能判断查询是否需要使用 RAG 搜索"""
        query_lower = query.lower().strip()
        
        # 如果查询很短（少于3个字符），通常不需要 RAG
        if len(query_lower) < 3:
            return False
        
        # 快速检查：明显的问候语
        greetings = ['你好', 'hi', 'hello', '再见', 'bye', '谢谢', 'thanks']
        if any(greeting in query_lower for greeting in greetings):
            return False
        
        # 快速检查：明显的代码相关问题
        code_indicators = [
            '这个函数', '这个类', '这个方法', '这个文件', '这个代码',
            '代码中', '项目中', '文件里', '函数里', '类里',
            '查找', '搜索', '找到', '定位', '位置', '行号',
            '错误', 'bug', '问题', '异常', '修复'
        ]
        if any(indicator in query_lower for indicator in code_indicators):
            return True
        
        # 需要 LLM 判断的模糊情况
        ambiguous_indicators = [
            '如何实现', '如何运行', '如何调试', '如何修复', '如何优化',
            '这个功能', '这个程序', '这个项目', '这个系统'
        ]
        if any(indicator in query_lower for indicator in ambiguous_indicators):
            # 这些情况需要 LLM 判断
            pass
        
        # 快速检查：文件扩展名
        if any(ext in query_lower for ext in ['.py', '.js', '.java', '.cpp', '.c', '.h', '.ts', '.vue', '.go', '.rs']):
            return True
        
        # 快速检查：编程语法
        if any(char in query_lower for char in ['(', ')', '{', '}', '[', ']', ';', ':', '=', '==', '!=']):
            return True
        
        # 使用 LLM 进行智能判断
        return self._llm_judge_rag_need(query)
    
    def _llm_judge_rag_need(self, query: str) -> bool:
        """使用 LLM 判断是否需要 RAG"""
        prompt = f"""你是一个智能助手，需要判断用户的问题是否需要搜索当前代码库来回答。

用户问题: {query}

请仔细分析这个问题是否需要搜索当前代码库中的具体代码、函数、类、文件或项目相关内容来回答。

判断标准：
1. 回答 "RAG" 如果问题询问：
   - 当前代码库中的具体函数、类、方法、文件
   - 当前项目的结构、配置、依赖
   - 当前代码中的错误、bug、问题
   - 当前项目的运行方式、部署方式
   - 当前代码库中的具体实现细节
   - 当前项目的功能、特性

2. 回答 "DIRECT" 如果问题询问：
   - 通用编程概念、理论、原理
   - 通用编程技能、学习方法
   - 通用技术知识、概念解释
   - 与当前代码库无关的一般性问题

特别注意：
- "如何实现这个功能？" 如果指当前项目的功能，回答 "RAG"
- "如何运行这个程序？" 如果指当前项目，回答 "RAG"
- "如何调试代码？" 如果是通用技能，回答 "DIRECT"

只回答 "RAG" 或 "DIRECT"，不要其他内容。"""

        try:
            response = self.client.generate(prompt, temperature=0.1, max_tokens=10)
            response = response.strip().upper()
            
            # 解析响应
            if 'RAG' in response:
                console.print(f"[dim]🤖 LLM 判断: RAG (需要搜索代码库) - 响应: '{response}'[/dim]")
                return True
            elif 'DIRECT' in response:
                console.print(f"[dim]🤖 LLM 判断: DIRECT (直接回答) - 响应: '{response}'[/dim]")
                return False
            else:
                # 如果 LLM 回答不明确，使用保守策略
                console.print(f"[dim]🤖 LLM 回答不明确: '{response}'，使用保守策略[/dim]")
                return False
                
        except Exception as e:
            # 如果 LLM 调用失败，使用保守策略
            console.print(f"[yellow]LLM 判断失败，使用保守策略: {str(e)}[/yellow]")
            return False
    
    def chat_direct(self, query: str) -> str:
        """直接回答，不使用 RAG"""
        prompt = f"""你是一个友好的AI助手。请用中文回答用户的问题。

用户问题: {query}

请提供有用、准确的回答。如果问题涉及编程或技术，请提供一般性的指导和建议。"""
        
        try:
            response = self.client.generate(prompt, temperature=0.7, max_tokens=1500)
            return response
        except Exception as e:
            return f"[red]错误: {str(e)}[/red]"
    
    def explain_code_rag(self, query: str, n_results: int = 10) -> str:
        """使用 RAG 解释代码"""
        console.print(f"\n[bold cyan]正在搜索相关代码...[/bold cyan]")
        
        # 搜索相关代码片段
        results = self.rag.search(query, n_results=n_results)
        
        if not results:
            return "[red]未找到相关代码[/red]"
        
        # 构建上下文
        context_parts = []
        included_files = set()
        
        console.print(f"\n[cyan]找到 {len(results)} 个相关代码片段:[/cyan]")
        
        for i, result in enumerate(results[:5]):  # 只显示前5个
            metadata = result['metadata']
            file_path = metadata['file_path']
            included_files.add(file_path)
            
            # 显示搜索结果
            console.print(f"  • {Path(file_path).name}:{metadata['start_line']}-{metadata['end_line']}", end="")
            if metadata.get('name'):
                console.print(f" [{metadata['name']}]", end="")
            console.print(f" (相关度: {1 - result['distance']:.2f})")
            
            # 构建上下文
            context_part = f"\n--- 文件: {file_path} (行 {metadata['start_line']}-{metadata['end_line']}) ---\n"
            if metadata.get('name'):
                context_part += f"名称: {metadata['name']}\n"
            if metadata.get('docstring'):
                context_part += f"说明: {metadata['docstring']}\n"
            context_part += f"\n{result['content']}\n"
            
            context_parts.append(context_part)
        
        context = '\n'.join(context_parts)
        
        # 构建 prompt
        prompt = f"""你是一个代码分析专家。请基于以下相关代码片段回答用户的问题。

用户问题: {query}

相关代码片段:
{context}

请提供详细、准确的解释。如果涉及具体的代码实现，请引用相关的函数或类名。用中文回答。"""
        
        # 调用 Ollama
        console.print(f"\n[bold cyan]正在生成解释...[/bold cyan]")
        
        try:
            response = self.client.generate(prompt, temperature=0.3, max_tokens=2000)
            return response
        except Exception as e:
            return f"[red]错误: {str(e)}[/red]"
    
    def modify_code(self, file_path: Path, instruction: str) -> Tuple[str, str]:
        """修改代码文件（使用 RAG 增强上下文）"""
        console.print(f"\n[bold cyan]正在读取文件: {file_path}[/bold cyan]")
        
        try:
            original_content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            return "", f"[red]错误: 无法读取文件 - {str(e)}[/red]"
        
        # 搜索相关代码以获得更好的上下文
        console.print("[cyan]搜索相关代码上下文...[/cyan]")
        context_query = f"{file_path.name} {instruction}"
        related_results = self.rag.search(context_query, n_results=5)
        
        # 构建额外上下文
        additional_context = ""
        if related_results:
            additional_context = "\n相关代码参考:\n"
            for result in related_results[:3]:
                if result['metadata']['file_path'] != str(file_path):
                    additional_context += f"\n--- {result['metadata']['file_path']} ---\n"
                    additional_context += result['content'] + "\n"
        
        # 获取文件扩展名对应的语言
        lang = file_path.suffix[1:] if file_path.suffix else ''
        
        # 构建 prompt
        prompt = f"""你是一个专业的程序员。请根据用户的需求修改以下代码。

原始代码文件 ({file_path.name}):
```{lang}
{original_content}
```

{additional_context}

修改需求: {instruction}

请返回修改后的完整代码。只返回代码内容，不要包含markdown代码块标记，不要包含额外的解释。"""
        
        console.print(f"\n[bold cyan]正在生成修改...[/bold cyan]")
        
        try:
            response = self.client.generate(prompt, temperature=0.2, max_tokens=4000)
            
            # 清理返回的内容
            lines = response.strip().split('\n')
            
            # 去除开头的 ```
            if lines and lines[0].strip().startswith('```'):
                lines = lines[1:]
            
            # 去除结尾的 ```
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            
            modified_content = '\n'.join(lines)
            
            # 更新索引
            if modified_content != original_content:
                self.rag.index_file(str(file_path), modified_content)
                self.rag._save_file_hashes()
            
            return modified_content, ""
        except Exception as e:
            return "", f"[red]错误: {str(e)}[/red]"
    
    def list_indexed_files(self) -> None:
        """列出已索引的文件"""
        stats = self.rag.get_stats()
        
        if not stats['indexed_files']:
            console.print("[yellow]没有已索引的文件[/yellow]")
            return
        
        # 创建表格
        table = Table(title="已索引文件")
        table.add_column("文件路径", style="cyan")
        table.add_column("代码块数", justify="right", style="green")
        
        # 统计每个文件的块数
        file_chunks = {}
        all_results = self.rag.collection.get(limit=stats['total_chunks'])
        
        for metadata in all_results['metadatas']:
            file_path = metadata['file_path']
            file_chunks[file_path] = file_chunks.get(file_path, 0) + 1
        
        # 排序并显示
        for file_path in sorted(file_chunks.keys()):
            table.add_row(file_path, str(file_chunks[file_path]))
        
        console.print(table)
        console.print(f"\n[bold]总计: {stats['total_files']} 个文件, {stats['total_chunks']} 个代码块[/bold]")


# CLI 部分
@click.group()
@click.option('--model', '-m', default='devstral:latest', help='Ollama 模型名称')
@click.option('--base-url', default='http://localhost:11434', help='Ollama API 地址')
@click.option('--index-dir', default='.codebase_index', help='索引存储目录')
@click.pass_context
def cli(ctx, model, base_url, index_dir):
    """Codebase Agent RAG - 基于 Ollama 和向量检索的智能代码库助手"""
    ctx.obj = CodebaseAgentRAG(model=model, base_url=base_url, index_dir=index_dir)


@cli.command()
@click.option('--path', '-p', default='.', help='代码库路径')
@click.option('--force', '-f', is_flag=True, help='强制重新索引所有文件')
@click.pass_obj
def index(agent, path, force):
    """建立或更新代码库索引"""
    path = Path(path).resolve()
    
    if not path.exists():
        console.print(f"[red]错误: 路径不存在 - {path}[/red]")
        return
    
    if force:
        console.print("[yellow]清除现有索引...[/yellow]")
        agent.rag.clear_index()
    
    agent.index_codebase(path)


@cli.command()
@click.argument('query')
@click.option('--results', '-n', default=10, help='返回结果数量')
@click.pass_obj
def search(agent, query, results):
    """搜索代码库"""
    console.print(f"\n[bold cyan]搜索: {query}[/bold cyan]")
    
    search_results = agent.rag.search(query, n_results=results)
    
    if not search_results:
        console.print("[yellow]未找到相关代码[/yellow]")
        return
    
    for i, result in enumerate(search_results):
        metadata = result['metadata']
        
        console.print(f"\n[bold green]结果 {i+1}:[/bold green]")
        console.print(f"文件: {metadata['file_path']}")
        console.print(f"位置: 行 {metadata['start_line']}-{metadata['end_line']}")
        
        if metadata.get('chunk_type'):
            console.print(f"类型: {metadata['chunk_type']}")
        if metadata.get('name'):
            console.print(f"名称: {metadata['name']}")
        
        console.print(f"相关度: {1 - result['distance']:.2f}")
        
        # 显示代码片段
        syntax = Syntax(result['content'][:200] + "..." if len(result['content']) > 200 else result['content'],
                       Path(metadata['file_path']).suffix[1:] or "text",
                       theme="monokai", line_numbers=False)
        console.print(Panel(syntax, border_style="dim"))


@cli.command()
@click.argument('query')
@click.option('--results', '-n', default=10, help='使用的搜索结果数量')
@click.pass_obj
def explain(agent, query, results):
    """使用 RAG 解释代码功能"""
    # 检查是否有索引
    stats = agent.rag.get_stats()
    if stats['total_chunks'] == 0:
        console.print("[yellow]警告: 没有索引的代码。请先运行 'index' 命令。[/yellow]")
        return
    
    result = agent.explain_code_rag(query, n_results=results)
    
    console.print("\n")
    console.print(Panel(result, title="[bold green]代码解释[/bold green]", 
                       title_align="left", border_style="green"))


@cli.command()
@click.argument('file_path')
@click.argument('instruction')
@click.option('--output', '-o', help='输出文件路径（默认覆盖原文件）')
@click.option('--dry-run', is_flag=True, help='只显示修改，不写入文件')
@click.pass_obj
def modify(agent, file_path, instruction, output, dry_run):
    """修改代码文件（使用 RAG 增强）"""
    file_path = Path(file_path).resolve()
    
    if not file_path.exists():
        console.print(f"[red]错误: 文件不存在 - {file_path}[/red]")
        return
    
    modified_content, error = agent.modify_code(file_path, instruction)
    
    if error:
        console.print(error)
        return
    
    # 显示修改后的代码
    console.print("\n")
    syntax = Syntax(modified_content, file_path.suffix[1:] if file_path.suffix else "text",
                   theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title=f"[bold green]修改后的代码 - {file_path.name}[/bold green]",
                       title_align="left", border_style="green"))
    
    if not dry_run:
        output_path = Path(output) if output else file_path
        
        # 备份原文件
        if output_path == file_path and file_path.exists():
            backup_path = file_path.with_suffix(file_path.suffix + '.bak')
            file_path.rename(backup_path)
            console.print(f"\n[yellow]原文件已备份到: {backup_path}[/yellow]")
        
        # 写入新文件
        output_path.write_text(modified_content, encoding='utf-8')
        console.print(f"[green]✓ 文件已保存到: {output_path}[/green]")
    else:
        console.print("\n[yellow]试运行模式 - 文件未被修改[/yellow]")


@cli.command()
@click.pass_obj
def stats(agent):
    """显示索引统计信息"""
    stats = agent.rag.get_stats()
    
    console.print(Panel.fit(
        f"[bold cyan]索引统计信息[/bold cyan]\n\n"
        f"总文件数: {stats['total_files']}\n"
        f"总代码块: {stats['total_chunks']}\n",
        border_style="cyan"
    ))
    
    if stats['chunk_types']:
        console.print("\n[bold]代码块类型分布:[/bold]")
        for chunk_type, count in sorted(stats['chunk_types'].items()):
            console.print(f"  {chunk_type}: {count}")
    
    console.print("\n[bold]使用以下命令查看详细文件列表:[/bold]")
    console.print("  codebase-agent-rag list-files")


@cli.command()
@click.pass_obj
def list_files(agent):
    """列出已索引的文件"""
    agent.list_indexed_files()


@cli.command()
@click.option('--yes', '-y', is_flag=True, help='跳过确认')
@click.pass_obj
def clear(agent, yes):
    """清除所有索引"""
    if not yes:
        if not click.confirm("[yellow]确定要清除所有索引吗？[/yellow]"):
            console.print("[cyan]已取消[/cyan]")
            return
    
    agent.rag.clear_index()
    console.print("[green]索引已清除[/green]")


@cli.command()
@click.argument('query')
@click.pass_obj
def test_rag_decision(agent, query):
    """测试 RAG 决策功能"""
    console.print(f"\n[bold cyan]查询: {query}[/bold cyan]")
    console.print("[dim]正在分析...[/dim]")
    
    use_rag = agent.should_use_rag(query)
    
    console.print(f"[bold]RAG 决策: {'🔍 使用 RAG' if use_rag else '💬 直接回答'}[/bold]")
    
    if use_rag:
        console.print("[yellow]原因: 检测到代码相关问题[/yellow]")
    else:
        console.print("[yellow]原因: 检测到一般性问题[/yellow]")


@cli.command()
@click.argument('query')
@click.pass_obj
def debug_llm_judgment(agent, query):
    """调试 LLM 判断功能"""
    console.print(f"\n[bold cyan]调试 LLM 判断[/bold cyan]")
    console.print(f"查询: {query}")
    console.print("=" * 50)
    
    # 直接调用 LLM 判断
    result = agent._llm_judge_rag_need(query)
    
    console.print(f"\n[bold]最终结果: {'🔍 RAG' if result else '💬 DIRECT'}[/bold]")


@cli.command()
@click.option('--results', '-n', default=10, help='使用的搜索结果数量')
@click.pass_obj
def chat(agent, results):
    """交互式 RAG 分析模式"""
    # 检查是否有索引
    stats = agent.rag.get_stats()
    if stats['total_chunks'] == 0:
        console.print("[yellow]警告: 没有索引的代码。请先运行 'index' 命令。[/yellow]")
        if not click.confirm("继续使用交互模式？"):
            return
    
    console.print(Panel.fit(
        f"[bold cyan]Codebase Agent RAG - 交互式分析模式[/bold cyan]\n"
        f"[yellow]索引文件数: {stats['total_files']}[/yellow]\n"
        f"[yellow]代码块数: {stats['total_chunks']}[/yellow]\n"
        f"[green]使用模型: {agent.model}[/green]\n"
        f"[dim]输入 'exit' 或 'quit' 退出[/dim]",
        border_style="cyan"
    ))
    
    # 强制模式标志
    force_rag = False
    force_direct = False
    
    while True:
        try:
            query = console.input("\n[bold green]?[/bold green] ")
            
            if query.lower() in ['exit', 'quit', 'q']:
                console.print("[yellow]再见！[/yellow]")
                break
            
            if not query.strip():
                continue
            
            # 特殊命令
            if query.startswith('/'):
                command = query[1:].strip().lower()
                
                if command == 'stats':
                    stats = agent.rag.get_stats()
                    console.print(f"文件数: {stats['total_files']}, 代码块: {stats['total_chunks']}")
                elif command == 'help':
                    console.print("[cyan]可用命令:[/cyan]")
                    console.print("  /stats - 显示索引统计")
                    console.print("  /help - 显示帮助")
                    console.print("  /clear - 清屏")
                    console.print("  /rag - 强制使用 RAG 搜索")
                    console.print("  /direct - 强制直接回答")
                elif command == 'clear':
                    console.clear()
                elif command == 'rag':
                    # 强制使用 RAG 搜索下一个查询
                    console.print("[yellow]已设置强制使用 RAG 搜索模式[/yellow]")
                    force_rag = True
                    continue
                elif command == 'direct':
                    # 强制直接回答下一个查询
                    console.print("[yellow]已设置强制直接回答模式[/yellow]")
                    force_direct = True
                    continue
                else:
                    console.print(f"[red]未知命令: {command}[/red]")
                
                continue
            
            # 智能判断是否需要使用 RAG
            if force_rag:
                console.print("[cyan]🔍 强制使用 RAG 搜索...[/cyan]")
                result = agent.explain_code_rag(query, n_results=results)
                force_rag = False  # 重置强制模式
            elif force_direct:
                console.print("[cyan]💬 强制直接回答...[/cyan]")
                result = agent.chat_direct(query)
                force_direct = False  # 重置强制模式
            else:
                console.print("[dim]🤔 正在分析查询类型...[/dim]")
                use_rag = agent.should_use_rag(query)
                if use_rag:
                    console.print("[cyan]🔍 检测到代码相关问题，使用 RAG 搜索...[/cyan]")
                    result = agent.explain_code_rag(query, n_results=results)
                else:
                    console.print("[cyan]💬 检测到一般性问题，直接回答...[/cyan]")
                    result = agent.chat_direct(query)
            
            console.print("\n")
            console.print(Panel(result, title="[bold green]回答[/bold green]",
                              title_align="left", border_style="green"))
            
        except KeyboardInterrupt:
            console.print("\n[yellow]已取消[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]错误: {str(e)}[/red]")


if __name__ == '__main__':
    cli()