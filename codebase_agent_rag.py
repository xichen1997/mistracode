#!/usr/bin/env python3
"""
Codebase Agent with RAG - Code explanation and modification tool using Ollama and vector database
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
    """Ollama API client"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "devstral:24b"):
        self.base_url = base_url
        self.model = model
        
    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 4000) -> str:
        """Call Ollama to generate response"""
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
            raise Exception("Unable to connect to Ollama service. Please ensure Ollama is running.")
        except Exception as e:
            raise Exception(f"Ollama API call failed: {str(e)}")
    
    def embeddings(self, text: str) -> List[float]:
        """Get text embedding vectors"""
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
            # If Ollama doesn't support embedding, return None
            return None
    
    def list_models(self) -> List[str]:
        """List available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except:
            return []

class CodeChunk:
    """Code chunk class"""
    
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
    """Code parser that breaks code files into meaningful chunks"""
    
    @staticmethod
    def parse_python(content: str, file_path: str) -> List[CodeChunk]:
        """Parse Python code"""
        chunks = []
        
        try:
            tree = ast.parse(content)
            
            # Extract functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    start_line = node.lineno
                    end_line = node.end_lineno or start_line
                    
                    # Get function code
                    lines = content.splitlines()
                    func_lines = lines[start_line-1:end_line]
                    func_content = '\n'.join(func_lines)
                    
                    # Extract docstring
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
                    
                    # Get class code
                    lines = content.splitlines()
                    class_lines = lines[start_line-1:end_line]
                    class_content = '\n'.join(class_lines)
                    
                    # Extract docstring
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
            # If parsing fails, return the entire file as one block
            pass
        
        # If no functions or classes found, or parsing failed, split the entire file into smaller blocks
        if not chunks:
            lines = content.splitlines()
            chunk_size = 50  # Number of lines per block
            
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
        """Parse JavaScript/TypeScript code (simplified version)"""
        chunks = []
        lines = content.splitlines()
        
        # Simple function and class detection
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Detect functions
            if ('function ' in line or 'const ' in line or 'let ' in line or 'var ' in line) and ('=' in line or '(' in line):
                start = i
                brace_count = 0
                found_brace = False
                
                # Find function end
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
        
        # If no functions found, process in chunks
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
        """Generic code parsing (chunked by line count)"""
        chunks = []
        lines = content.splitlines()
        chunk_size = 50  # Number of lines per block
        
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
        """Parse code based on file type"""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.py':
            return CodeParser.parse_python(content, file_path)
        elif ext in ['.js', '.jsx', '.ts', '.tsx']:
            return CodeParser.parse_javascript(content, file_path)
        else:
            return CodeParser.parse_generic(content, file_path)

class CodebaseRAG:
    """RAG-based codebase management system"""
    
    def __init__(self, persist_dir: str = ".codebase_index", ollama_client: OllamaClient = None):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)
        
        # Initialize embedding model
        if ollama_client:
            # Use Ollama for embedding
            console.print("[cyan]Using Ollama for embedding...[/cyan]")
            self.embedder = ollama_client
            self.use_ollama = True
        else:
            # Fall back to SentenceTransformer
            console.print("[cyan]Loading embedding model...[/cyan]")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_ollama = False
        
        # Initialize vector database
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir / "chroma"),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or load collection
        try:
            self.collection = self.chroma_client.get_collection("codebase")
            console.print("[green]Loading existing index[/green]")
        except:
            self.collection = self.chroma_client.create_collection(
                name="codebase",
                metadata={"hnsw:space": "cosine"}
            )
            console.print("[yellow]Creating new index[/yellow]")
        
        # File hash cache (for detecting file changes)
        self.hash_cache_file = self.persist_dir / "file_hashes.json"
        self.file_hashes = self._load_file_hashes()
    
    def _load_file_hashes(self) -> Dict[str, str]:
        """Load file hash cache"""
        if self.hash_cache_file.exists():
            with open(self.hash_cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_file_hashes(self):
        """Save file hash cache"""
        with open(self.hash_cache_file, 'w') as f:
            json.dump(self.file_hashes, f)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate file hash"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _embed_text(self, text: str) -> List[float]:
        """Generate text embedding"""
        if self.use_ollama:
            # Use Ollama for embedding
            embedding = self.embedder.embeddings(text)
            if embedding is None:
                raise Exception("Ollama model does not support embedding functionality")
            return embedding
        else:
            # Use SentenceTransformer for embedding
            return self.embedder.encode(text).tolist()
    
    def index_file(self, file_path: str, content: str):
        """Index a single file"""
        # Check if file has changed
        current_hash = self._get_file_hash(file_path)
        
        if file_path in self.file_hashes and self.file_hashes[file_path] == current_hash:
            return  # File unchanged, skip
        
        # Delete old index
        try:
            results = self.collection.get(where={"file_path": file_path})
            if results['ids']:
                self.collection.delete(ids=results['ids'])
        except:
            pass
        
        # Parse code into chunks
        chunks = CodeParser.parse_file(content, file_path)
        
        # Index each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_path}:{chunk.start_line}:{chunk.end_line}"
            
            # Prepare index content
            index_content = f"File: {file_path}\n"
            if chunk.metadata.get('name'):
                index_content += f"Name: {chunk.metadata['name']}\n"
            if chunk.metadata.get('docstring'):
                index_content += f"Description: {chunk.metadata['docstring']}\n"
            index_content += f"\n{chunk.content}"
            
            # Generate embedding
            embedding = self._embed_text(index_content)
            
            # Store to vector database
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
        
        # Update file hash
        self.file_hashes[file_path] = current_hash
    
    def search(self, query: str, n_results: int = 10) -> List[Dict]:
        """Search for related code snippets"""
        # Generate query embedding
        query_embedding = self._embed_text(query)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
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
        """Get index statistics"""
        count = self.collection.count()
        
        # Get metadata of all documents for statistics
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
        """Clear all indices"""
        self.chroma_client.delete_collection("codebase")
        self.collection = self.chroma_client.create_collection(
            name="codebase",
            metadata={"hnsw:space": "cosine"}
        )
        self.file_hashes = {}
        self._save_file_hashes()

class CodebaseAgentRAG:
    """RAG-enabled intelligent codebase assistant"""
    
    def __init__(self, model: str = "devstral:24b", 
                 base_url: str = "http://localhost:11434",
                 index_dir: str = ".codebase_index"):
        self.client = OllamaClient(base_url=base_url, model=model)
        self.model = model
        self.rag = CodebaseRAG(persist_dir=index_dir, ollama_client=self.client)
        
        # Supported file extensions
        self.supported_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', 
            '.h', '.hpp', '.cs', '.go', '.rs', '.php', '.rb', '.swift',
            '.kt', '.scala', '.r', '.m', '.mm', '.vue', '.dart', '.lua',
            '.sh', '.bash', '.zsh', '.fish', '.ps1', '.yaml', '.yml',
            '.json', '.xml', '.toml', '.ini', '.cfg', '.conf', '.md',
            '.txt', '.sql', '.dockerfile', '.makefile'
        }
        
        # Ignored directories
        self.ignore_dirs = {
            '.git', '__pycache__', 'node_modules', '.venv', 'venv', 'venv_rag',
            'env', 'env_rag', 'virtualenv', 'conda', 'miniconda', 'anaconda',
            'dist', 'build', '.idea', '.vscode', 'target',
            'bin', 'obj', '.pytest_cache', '.mypy_cache', '.tox',
            'coverage', '.coverage', 'htmlcov', '.sass-cache',
            '.codebase_index',  # Ignore index directory
            'site-packages', 'lib64', 'include', 'share',  # Common virtual environment directories
            '.DS_Store', 'Thumbs.db'  # System files
        }
        
        # Check Ollama connection
        self._check_ollama_connection()
    
    def _check_ollama_connection(self):
        """Check if Ollama service is available"""
        try:
            models = self.client.list_models()
            if not models:
                console.print("[yellow]Warning: No models installed in Ollama[/yellow]")
                console.print("[cyan]Please run: ollama pull devstral:24b[/cyan]")
            elif self.model not in models:
                console.print(f"[yellow]Warning: Model {self.model} not installed[/yellow]")
                console.print(f"[cyan]Available models: {', '.join(models)}[/cyan]")
                console.print(f"[cyan]Install model: ollama pull {self.model}[/cyan]")
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            console.print("[cyan]Please ensure Ollama is running: ollama serve[/cyan]")
            exit(1)
    
    def scan_directory(self, path: Path, show_progress: bool = True) -> List[Dict]:
        """Scan directory to get all code files"""
        files = []
        
        # Collect all files that need processing
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
                task = progress.add_task("[cyan]Scanning code files...", total=len(all_files))
                
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
                        console.print(f"[yellow]Warning: Unable to read file {file_path}: {e}[/yellow]")
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
        """Index the entire codebase"""
        console.print(f"\n[bold cyan]Indexing codebase...[/bold cyan]")
        
        files = self.scan_directory(path)
        
        if not files:
            console.print("[red]No code files found[/red]")
            return
        
        console.print(f"[green]Found {len(files)} code files[/green]")
        
        # Index files
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Building index...", total=len(files))
            
            for file in files:
                try:
                    self.rag.index_file(file['full_path'], file['content'])
                    progress.update(task, advance=1)
                except Exception as e:
                    console.print(f"[yellow]Index failed {file['path']}: {e}[/yellow]")
        
        # Save file hashes
        self.rag._save_file_hashes()
        
        # Display statistics
        stats = self.rag.get_stats()
        console.print(f"\n[green]Indexing complete![/green]")
        console.print(f"Total files: {stats['total_files']}")
        console.print(f"Total code blocks: {stats['total_chunks']}")
        
        if stats['chunk_types']:
            console.print("\nCode block type distribution:")
            for chunk_type, count in stats['chunk_types'].items():
                console.print(f"  {chunk_type}: {count}")
    
    def should_use_rag(self, query: str) -> bool:
        """Intelligently determine if query needs RAG search"""
        query_lower = query.lower().strip()
        
        # If query is very short (less than 3 characters), usually doesn't need RAG
        if len(query_lower) < 3:
            return False
        
        # Quick check: obvious greetings
        greetings = ['hello', 'hi', 'hello', 'goodbye', 'bye', 'thanks', 'thanks']
        if any(greeting in query_lower for greeting in greetings):
            return False
        
        # Quick check: obvious code-related questions
        code_indicators = [
            'this function', 'this class', 'this method', 'this file', 'this code',
            'in code', 'in project', 'in file', 'in function', 'in class',
            'find', 'search', 'locate', 'position', 'line number',
            'error', 'bug', 'issue', 'exception', 'fix'
        ]
        if any(indicator in query_lower for indicator in code_indicators):
            return True
        
        # Ambiguous cases that need LLM judgment
        ambiguous_indicators = [
            'how to implement', 'how to run', 'how to debug', 'how to fix', 'how to optimize',
            'this feature', 'this program', 'this project', 'this system'
        ]
        if any(indicator in query_lower for indicator in ambiguous_indicators):
            # These cases need LLM judgment
            pass
        
        # Quick check: file extensions
        if any(ext in query_lower for ext in ['.py', '.js', '.java', '.cpp', '.c', '.h', '.ts', '.vue', '.go', '.rs']):
            return True
        
        # Quick check: programming syntax
        if any(char in query_lower for char in ['(', ')', '{', '}', '[', ']', ';', ':', '=', '==', '!=']):
            return True
        
        # Use LLM for intelligent judgment
        return self._llm_judge_rag_need(query)
    
    def _llm_judge_rag_need(self, query: str) -> bool:
        """Use LLM to judge whether RAG is needed"""
        prompt = f"""You are an intelligent assistant that needs to determine whether a user's question requires searching the current codebase to answer.

User question: {query}

Please carefully analyze whether this question needs to search for specific code, functions, classes, files, or project-related content in the current codebase to answer.

Judgment criteria:
1. Answer "RAG" if the question asks about:
   - Specific functions, classes, methods, files in the current codebase
   - Current project structure, configuration, dependencies
   - Errors, bugs, issues in the current code
   - How the current project runs or is deployed
   - Specific implementation details in the current codebase
   - Features and characteristics of the current project

2. Answer "DIRECT" if the question asks about:
   - General programming concepts, theories, principles
   - General programming skills, learning methods
   - General technical knowledge, concept explanations
   - General questions unrelated to the current codebase

Special attention:
- "How to implement this feature?" If referring to current project features, answer "RAG"
- "How to run this program?" If referring to current project, answer "RAG"
- "How to debug code?" If it's a general skill, answer "DIRECT"

Only answer "RAG" or "DIRECT", no other content."""

        try:
            response = self.client.generate(prompt, temperature=0.1, max_tokens=10)
            response = response.strip().upper()
            
            # Parse response
            if 'RAG' in response:
                console.print(f"[dim]🤖 LLM judgment: RAG (need to search codebase) - response: '{response}'[/dim]")
                return True
            elif 'DIRECT' in response:
                console.print(f"[dim]🤖 LLM judgment: DIRECT (direct answer) - response: '{response}'[/dim]")
                return False
            else:
                # If LLM answer is unclear, use conservative strategy
                console.print(f"[dim]🤖 LLM answer unclear: '{response}', using conservative strategy[/dim]")
                return False
                
        except Exception as e:
            # If LLM call fails, use conservative strategy
            console.print(f"[yellow]LLM judgment failed, using conservative strategy: {str(e)}[/yellow]")
            return False
    
    def chat_direct(self, query: str) -> str:
        """Direct answer without using RAG"""
        prompt = f"""You are a friendly AI assistant. Please answer the user's question in English.

User question: {query}

Please provide helpful and accurate answers. If the question involves programming or technology, please provide general guidance and advice."""
        
        try:
            response = self.client.generate(prompt, temperature=0.7, max_tokens=1500)
            return response
        except Exception as e:
            return f"[red]Error: {str(e)}[/red]"
    
    def explain_code_rag(self, query: str, n_results: int = 10) -> str:
        """Use RAG to explain code"""
        console.print(f"\n[bold cyan]Searching for relevant code...[/bold cyan]")
        
        # Search for relevant code snippets
        results = self.rag.search(query, n_results=n_results)
        
        if not results:
            return "[red]No relevant code found[/red]"
        
        # Build context
        context_parts = []
        included_files = set()
        
        console.print(f"\n[cyan]Found {len(results)} relevant code snippets:[/cyan]")
        
        for i, result in enumerate(results[:5]):  # Only show first 5
            metadata = result['metadata']
            file_path = metadata['file_path']
            included_files.add(file_path)
            
            # Display search results
            console.print(f"  • {Path(file_path).name}:{metadata['start_line']}-{metadata['end_line']}", end="")
            if metadata.get('name'):
                console.print(f" [{metadata['name']}]", end="")
            console.print(f" (Relevance: {1 - result['distance']:.2f})")
            
            # Build context
            context_part = f"\n--- File: {file_path} (Lines {metadata['start_line']}-{metadata['end_line']}) ---\n"
            if metadata.get('name'):
                context_part += f"Name: {metadata['name']}\n"
            if metadata.get('docstring'):
                context_part += f"Description: {metadata['docstring']}\n"
            context_part += f"\n{result['content']}\n"
            
            context_parts.append(context_part)
        
        context = '\n'.join(context_parts)
        
        # Build prompt
        prompt = f"""You are a code analysis expert. Please answer the user's question based on the following relevant code snippets.

User question: {query}

Relevant code snippets:
{context}

Please provide detailed and accurate explanations. If it involves specific code implementation, please reference the relevant function or class names. Answer in English."""
        
        # Call Ollama
        console.print(f"\n[bold cyan]Generating explanation...[/bold cyan]")
        
        try:
            response = self.client.generate(prompt, temperature=0.3, max_tokens=2000)
            return response
        except Exception as e:
            return f"[red]Error: {str(e)}[/red]"
    
    def modify_code(self, file_path: Path, instruction: str) -> Tuple[str, str]:
        """Modify code file (using RAG-enhanced context)"""
        console.print(f"\n[bold cyan]Reading file: {file_path}[/bold cyan]")
        
        try:
            original_content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            return "", f"[red]Error: Unable to read file - {str(e)}[/red]"
        
        # Search for related code to get better context
        console.print("[cyan]Searching for related code context...[/cyan]")
        context_query = f"{file_path.name} {instruction}"
        related_results = self.rag.search(context_query, n_results=5)
        
        # Build additional context
        additional_context = ""
        if related_results:
            additional_context = "\nRelated code reference:\n"
            for result in related_results[:3]:
                if result['metadata']['file_path'] != str(file_path):
                    additional_context += f"\n--- {result['metadata']['file_path']} ---\n"
                    additional_context += result['content'] + "\n"
        
        # Get language corresponding to file extension
        lang = file_path.suffix[1:] if file_path.suffix else ''
        
        # Build prompt
        prompt = f"""You are a professional programmer. Please modify the following code according to the user's requirements.

Original code file ({file_path.name}):
```{lang}
{original_content}
```

{additional_context}

Modification requirement: {instruction}

Please return the complete modified code. Only return the code content, do not include markdown code block markers, do not include additional explanations."""
        
        console.print(f"\n[bold cyan]Generating modifications...[/bold cyan]")
        
        try:
            response = self.client.generate(prompt, temperature=0.2, max_tokens=4000)
            
            # Clean up returned content
            lines = response.strip().split('\n')
            
            # Remove leading ```
            if lines and lines[0].strip().startswith('```'):
                lines = lines[1:]
            
            # Remove trailing ```
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            
            modified_content = '\n'.join(lines)
            
            # Update index
            if modified_content != original_content:
                self.rag.index_file(str(file_path), modified_content)
                self.rag._save_file_hashes()
            
            return modified_content, ""
        except Exception as e:
            return "", f"[red]Error: {str(e)}[/red]"
    
    def list_indexed_files(self) -> None:
        """List indexed files"""
        stats = self.rag.get_stats()
        
        if not stats['indexed_files']:
            console.print("[yellow]No indexed files[/yellow]")
            return
        
        # Create table
        table = Table(title="Indexed Files")
        table.add_column("File Path", style="cyan")
        table.add_column("Code Blocks", justify="right", style="green")
        
        # Count blocks per file
        file_chunks = {}
        all_results = self.rag.collection.get(limit=stats['total_chunks'])
        
        for metadata in all_results['metadatas']:
            file_path = metadata['file_path']
            file_chunks[file_path] = file_chunks.get(file_path, 0) + 1
        
        # Sort and display
        for file_path in sorted(file_chunks.keys()):
            table.add_row(file_path, str(file_chunks[file_path]))
        
        console.print(table)
        console.print(f"\n[bold]Total: {stats['total_files']} files, {stats['total_chunks']} code blocks[/bold]")


# CLI section
@click.group()
@click.option('--model', '-m', default='devstral:24b', help='Ollama model name')
@click.option('--base-url', default='http://localhost:11434', help='Ollama API address')
@click.option('--index-dir', default='.codebase_index', help='Index storage directory')
@click.pass_context
def cli(ctx, model, base_url, index_dir):
    """Codebase Agent RAG - Intelligent codebase assistant based on Ollama and vector retrieval"""
    ctx.obj = CodebaseAgentRAG(model=model, base_url=base_url, index_dir=index_dir)


@cli.command()
@click.option('--path', '-p', default='.', help='Codebase path')
@click.option('--force', '-f', is_flag=True, help='Force re-index all files')
@click.pass_obj
def index(agent, path, force):
    """Build or update codebase index"""
    path = Path(path).resolve()
    
    if not path.exists():
        console.print(f"[red]Error: Path does not exist - {path}[/red]")
        return
    
    if force:
        console.print("[yellow]Clearing existing index...[/yellow]")
        agent.rag.clear_index()
    
    agent.index_codebase(path)


@cli.command()
@click.argument('query')
@click.option('--results', '-n', default=10, help='Number of results to return')
@click.pass_obj
def search(agent, query, results):
    """Search codebase"""
    console.print(f"\n[bold cyan]Searching: {query}[/bold cyan]")
    
    search_results = agent.rag.search(query, n_results=results)
    
    if not search_results:
        console.print("[yellow]No relevant code found[/yellow]")
        return
    
    for i, result in enumerate(search_results):
        metadata = result['metadata']
        
        console.print(f"\n[bold green]Result {i+1}:[/bold green]")
        console.print(f"File: {metadata['file_path']}")
        console.print(f"Location: Lines {metadata['start_line']}-{metadata['end_line']}")
        
        if metadata.get('chunk_type'):
            console.print(f"Type: {metadata['chunk_type']}")
        if metadata.get('name'):
            console.print(f"Name: {metadata['name']}")
        
        console.print(f"Relevance: {1 - result['distance']:.2f}")
        
        # Display code snippet
        syntax = Syntax(result['content'][:200] + "..." if len(result['content']) > 200 else result['content'],
                       Path(metadata['file_path']).suffix[1:] or "text",
                       theme="monokai", line_numbers=False)
        console.print(Panel(syntax, border_style="dim"))


@cli.command()
@click.argument('query')
@click.option('--results', '-n', default=10, help='Number of search results to use')
@click.pass_obj
def explain(agent, query, results):
    """Use RAG to explain code functionality"""
    # Check if there is an index
    stats = agent.rag.get_stats()
    if stats['total_chunks'] == 0:
        console.print("[yellow]Warning: No indexed code. Please run 'index' command first.[/yellow]")
        return
    
    result = agent.explain_code_rag(query, n_results=results)
    
    console.print("\n")
    console.print(Panel(result, title="[bold green]Code Explanation[/bold green]", 
                       title_align="left", border_style="green"))


@cli.command()
@click.argument('file_path')
@click.argument('instruction')
@click.option('--output', '-o', help='Output file path (default overwrites original file)')
@click.option('--dry-run', is_flag=True, help='Only display changes, do not write to file')
@click.pass_obj
def modify(agent, file_path, instruction, output, dry_run):
    """Modify code file (RAG-enhanced)"""
    file_path = Path(file_path).resolve()
    
    if not file_path.exists():
        console.print(f"[red]Error: File does not exist - {file_path}[/red]")
        return
    
    modified_content, error = agent.modify_code(file_path, instruction)
    
    if error:
        console.print(error)
        return
    
    # Display modified code
    console.print("\n")
    syntax = Syntax(modified_content, file_path.suffix[1:] if file_path.suffix else "text",
                   theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title=f"[bold green]Modified Code - {file_path.name}[/bold green]",
                       title_align="left", border_style="green"))
    
    if not dry_run:
        output_path = Path(output) if output else file_path
        
        # Backup original file
        if output_path == file_path and file_path.exists():
            backup_path = file_path.with_suffix(file_path.suffix + '.bak')
            file_path.rename(backup_path)
            console.print(f"\n[yellow]Original file backed up to: {backup_path}[/yellow]")
        
        # Write new file
        output_path.write_text(modified_content, encoding='utf-8')
        console.print(f"[green]✓ File saved to: {output_path}[/green]")
    else:
        console.print("\n[yellow]Dry run mode - File not modified[/yellow]")


@cli.command()
@click.pass_obj
def stats(agent):
    """Display index statistics"""
    stats = agent.rag.get_stats()
    
    console.print(Panel.fit(
        f"[bold cyan]Index Statistics[/bold cyan]\n\n"
        f"Total files: {stats['total_files']}\n"
        f"Total code blocks: {stats['total_chunks']}\n",
        border_style="cyan"
    ))
    
    if stats['chunk_types']:
        console.print("\n[bold]Code block type distribution:[/bold]")
        for chunk_type, count in sorted(stats['chunk_types'].items()):
            console.print(f"  {chunk_type}: {count}")
    
    console.print("\n[bold]Use the following command to view detailed file list:[/bold]")
    console.print("  codebase-agent-rag list-files")


@cli.command()
@click.pass_obj
def list_files(agent):
    """List indexed files"""
    agent.list_indexed_files()


@cli.command()
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation')
@click.pass_obj
def clear(agent, yes):
    """Clear all indexes"""
    if not yes:
        if not click.confirm("[yellow]Are you sure you want to clear all indexes?[/yellow]"):
            console.print("[cyan]Cancelled[/cyan]")
            return
    
    agent.rag.clear_index()
    console.print("[green]Index cleared[/green]")


@cli.command()
@click.argument('query')
@click.pass_obj
def test_rag_decision(agent, query):
    """Test RAG decision functionality"""
    console.print(f"\n[bold cyan]Query: {query}[/bold cyan]")
    console.print("[dim]Analyzing...[/dim]")
    
    use_rag = agent.should_use_rag(query)
    
    console.print(f"[bold]RAG decision: {'🔍 Use RAG' if use_rag else '💬 Direct answer'}[/bold]")
    
    if use_rag:
        console.print("[yellow]Reason: Code-related question detected[/yellow]")
    else:
        console.print("[yellow]Reason: General question detected[/yellow]")


@cli.command()
@click.argument('query')
@click.pass_obj
def debug_llm_judgment(agent, query):
    """Debug LLM judgment functionality"""
    console.print(f"\n[bold cyan]Debug LLM judgment[/bold cyan]")
    console.print(f"Query: {query}")
    console.print("=" * 50)
    
    # Direct call to LLM judgment
    result = agent._llm_judge_rag_need(query)
    
    console.print(f"\n[bold]Final result: {'🔍 RAG' if result else '💬 DIRECT'}[/bold]")


@cli.command()
@click.option('--results', '-n', default=10, help='Number of search results to use')
@click.pass_obj
def chat(agent, results):
    """Interactive RAG analysis mode"""
    # Check if there is an index
    stats = agent.rag.get_stats()
    if stats['total_chunks'] == 0:
        console.print("[yellow]Warning: No indexed code. Please run 'index' command first.[/yellow]")
        if not click.confirm("Continue with interactive mode?"):
            return
    
    console.print(Panel.fit(
        f"[bold cyan]Codebase Agent RAG - Interactive Analysis Mode[/bold cyan]\n"
        f"[yellow]Indexed files: {stats['total_files']}[/yellow]\n"
        f"[yellow]Code blocks: {stats['total_chunks']}[/yellow]\n"
        f"[green]Using model: {agent.model}[/green]\n"
        f"[dim]Enter 'exit' or 'quit' to exit[/dim]",
        border_style="cyan"
    ))
    
    # Force mode flags
    force_rag = False
    force_direct = False
    
    while True:
        try:
            query = console.input("\n[bold green]?[/bold green] ")
            
            if query.lower() in ['exit', 'quit', 'q']:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if not query.strip():
                continue
            
            # Special commands
            if query.startswith('/'):
                command = query[1:].strip().lower()
                
                if command == 'stats':
                    stats = agent.rag.get_stats()
                    console.print(f"Files: {stats['total_files']}, Code blocks: {stats['total_chunks']}")
                elif command == 'help':
                    console.print("[cyan]Available commands:[/cyan]")
                    console.print("  /stats - Show index statistics")
                    console.print("  /help - Show help")
                    console.print("  /clear - Clear screen")
                    console.print("  /rag - Force use RAG search")
                    console.print("  /direct - Force direct answer")
                elif command == 'clear':
                    console.clear()
                elif command == 'rag':
                    # Force use RAG search for next query
                    console.print("[yellow]Set to force RAG search mode[/yellow]")
                    force_rag = True
                    continue
                elif command == 'direct':
                    # Force direct answer for next query
                    console.print("[yellow]Set to force direct answer mode[/yellow]")
                    force_direct = True
                    continue
                else:
                    console.print(f"[red]Unknown command: {command}[/red]")
                
                continue
            
            # Intelligent judgment of whether RAG is needed
            if force_rag:
                console.print("[cyan]🔍 Force using RAG search...[/cyan]")
                result = agent.explain_code_rag(query, n_results=results)
                force_rag = False  # Reset force mode
            elif force_direct:
                console.print("[cyan]💬 Force direct answer...[/cyan]")
                result = agent.chat_direct(query)
                force_direct = False  # Reset force mode
            else:
                console.print("[dim]🤔 Analyzing query type...[/dim]")
                use_rag = agent.should_use_rag(query)
                if use_rag:
                    console.print("[cyan]🔍 Code-related question detected, using RAG search...[/cyan]")
                    result = agent.explain_code_rag(query, n_results=results)
                else:
                    console.print("[cyan]💬 General question detected, direct answer...[/cyan]")
                    result = agent.chat_direct(query)
            
            console.print("\n")
            console.print(Panel(result, title="[bold green]Answer[/bold green]",
                              title_align="left", border_style="green"))
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")


if __name__ == '__main__':
    cli()