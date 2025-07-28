#!/usr/bin/env python3
"""
Debug utilities for inspecting codebase index
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree
import click

from codebase_agent_rag import CodebaseAgentRAG

console = Console()

class IndexDebugger:
    """Debug utilities for codebase index"""
    
    def __init__(self, index_dir: str = ".codebase_index"):
        self.index_dir = Path(index_dir)
        self.agent = CodebaseAgentRAG(index_dir=index_dir)
    
    def inspect_collection(self):
        """检查 ChromaDB 集合的详细信息"""
        console.print("[bold cyan]ChromaDB 集合详细信息[/bold cyan]")
        
        try:
            collection = self.agent.rag.collection
            count = collection.count()
            
            console.print(f"集合名称: {collection.name}")
            console.print(f"文档总数: {count}")
            
            if count > 0:
                # 获取所有文档
                all_docs = collection.get(limit=count)
                
                console.print(f"IDs 数量: {len(all_docs['ids'])}")
                console.print(f"Documents 数量: {len(all_docs['documents'])}")
                console.print(f"Metadatas 数量: {len(all_docs['metadatas'])}")
                
                # 分析元数据字段
                if all_docs['metadatas']:
                    metadata_fields = set()
                    for metadata in all_docs['metadatas']:
                        metadata_fields.update(metadata.keys())
                    
                    console.print(f"元数据字段: {sorted(metadata_fields)}")
                
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
    
    def show_sample_documents(self, limit: int = 5):
        """显示样本文档"""
        console.print(f"[bold cyan]样本文档 (前 {limit} 个)[/bold cyan]")
        
        try:
            collection = self.agent.rag.collection
            count = collection.count()
            
            if count == 0:
                console.print("[yellow]没有文档[/yellow]")
                return
            
            sample_docs = collection.get(limit=min(limit, count))
            
            for i in range(len(sample_docs['ids'])):
                doc_id = sample_docs['ids'][i]
                content = sample_docs['documents'][i]
                metadata = sample_docs['metadatas'][i]
                
                console.print(f"\n[bold green]文档 {i+1}:[/bold green]")
                console.print(f"ID: {doc_id}")
                
                # 显示元数据
                metadata_table = Table(title="元数据")
                metadata_table.add_column("字段", style="cyan")
                metadata_table.add_column("值", style="yellow")
                
                for key, value in metadata.items():
                    metadata_table.add_row(key, str(value))
                
                console.print(metadata_table)
                
                # 显示内容预览
                preview = content[:200] + "..." if len(content) > 200 else content
                syntax = Syntax(preview, "python", theme="monokai", line_numbers=False)
                console.print(Panel(syntax, title="内容预览", border_style="dim"))
        
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
    
    def analyze_file_distribution(self):
        """分析文件分布"""
        console.print("[bold cyan]文件分布分析[/bold cyan]")
        
        try:
            collection = self.agent.rag.collection
            count = collection.count()
            
            if count == 0:
                console.print("[yellow]没有文档[/yellow]")
                return
            
            all_docs = collection.get(limit=count)
            
            # 统计文件分布
            file_chunks = {}
            chunk_types = {}
            
            for metadata in all_docs['metadatas']:
                file_path = metadata.get('file_path', 'unknown')
                chunk_type = metadata.get('chunk_type', 'unknown')
                
                file_chunks[file_path] = file_chunks.get(file_path, 0) + 1
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            # 显示文件分布
            file_table = Table(title="文件代码块分布")
            file_table.add_column("文件路径", style="cyan")
            file_table.add_column("代码块数", justify="right", style="green")
            
            for file_path, count in sorted(file_chunks.items(), key=lambda x: x[1], reverse=True):
                file_table.add_row(file_path, str(count))
            
            console.print(file_table)
            
            # 显示代码块类型分布
            type_table = Table(title="代码块类型分布")
            type_table.add_column("类型", style="cyan")
            type_table.add_column("数量", justify="right", style="green")
            
            for chunk_type, count in sorted(chunk_types.items(), key=lambda x: x[1], reverse=True):
                type_table.add_row(chunk_type, str(count))
            
            console.print(type_table)
            
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
    
    def search_debug(self, query: str, n_results: int = 5):
        """调试搜索功能"""
        console.print(f"[bold cyan]搜索调试: {query}[/bold cyan]")
        
        try:
            # 执行搜索
            results = self.agent.rag.search(query, n_results=n_results)
            
            if not results:
                console.print("[yellow]没有搜索结果[/yellow]")
                return
            
            console.print(f"找到 {len(results)} 个结果")
            
            for i, result in enumerate(results):
                console.print(f"\n[bold green]结果 {i+1}:[/bold green]")
                console.print(f"ID: {result['id']}")
                console.print(f"相关度分数: {1 - result['distance']:.4f}")
                console.print(f"距离: {result['distance']:.4f}")
                
                # 显示元数据
                metadata = result['metadata']
                console.print(f"文件: {metadata.get('file_path', 'N/A')}")
                console.print(f"行数: {metadata.get('start_line', 'N/A')}-{metadata.get('end_line', 'N/A')}")
                console.print(f"类型: {metadata.get('chunk_type', 'N/A')}")
                if metadata.get('name'):
                    console.print(f"名称: {metadata['name']}")
                
                # 显示内容预览
                content = result['content']
                preview = content[:150] + "..." if len(content) > 150 else content
                syntax = Syntax(preview, "python", theme="monokai", line_numbers=False)
                console.print(Panel(syntax, title="内容", border_style="dim"))
        
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
    
    def show_file_hashes(self):
        """显示文件哈希缓存"""
        console.print("[bold cyan]文件哈希缓存[/bold cyan]")
        
        hash_file = self.agent.rag.hash_cache_file
        
        if not hash_file.exists():
            console.print("[yellow]哈希缓存文件不存在[/yellow]")
            return
        
        try:
            with open(hash_file, 'r') as f:
                hashes = json.load(f)
            
            if not hashes:
                console.print("[yellow]哈希缓存为空[/yellow]")
                return
            
            hash_table = Table(title="文件哈希")
            hash_table.add_column("文件路径", style="cyan")
            hash_table.add_column("MD5 哈希", style="yellow")
            
            for file_path, file_hash in sorted(hashes.items()):
                hash_table.add_row(file_path, file_hash)
            
            console.print(hash_table)
            console.print(f"\n总计: {len(hashes)} 个文件")
            
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
    
    def validate_index_integrity(self):
        """验证索引完整性"""
        console.print("[bold cyan]索引完整性检查[/bold cyan]")
        
        try:
            collection = self.agent.rag.collection
            count = collection.count()
            
            if count == 0:
                console.print("[yellow]索引为空[/yellow]")
                return
            
            all_docs = collection.get(limit=count)
            
            issues = []
            
            # 检查必需字段
            required_fields = ['file_path', 'start_line', 'end_line', 'chunk_type']
            
            for i, metadata in enumerate(all_docs['metadatas']):
                doc_id = all_docs['ids'][i]
                
                for field in required_fields:
                    if field not in metadata:
                        issues.append(f"文档 {doc_id}: 缺少字段 '{field}'")
                
                # 检查行号是否合理
                start_line = metadata.get('start_line')
                end_line = metadata.get('end_line')
                
                if start_line and end_line:
                    if start_line > end_line:
                        issues.append(f"文档 {doc_id}: 起始行 {start_line} > 结束行 {end_line}")
                    if start_line < 1:
                        issues.append(f"文档 {doc_id}: 起始行 {start_line} < 1")
            
            if issues:
                console.print(f"[red]发现 {len(issues)} 个问题:[/red]")
                for issue in issues:
                    console.print(f"  • {issue}")
            else:
                console.print("[green]索引完整性检查通过[/green]")
                
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
    
    def export_index_data(self, output_file: str = "index_export.json"):
        """导出索引数据到 JSON 文件"""
        console.print(f"[bold cyan]导出索引数据到 {output_file}[/bold cyan]")
        
        try:
            collection = self.agent.rag.collection
            count = collection.count()
            
            if count == 0:
                console.print("[yellow]没有数据可导出[/yellow]")
                return
            
            all_docs = collection.get(limit=count)
            
            export_data = {
                'collection_name': collection.name,
                'total_documents': count,
                'documents': []
            }
            
            for i in range(len(all_docs['ids'])):
                doc_data = {
                    'id': all_docs['ids'][i],
                    'content': all_docs['documents'][i],
                    'metadata': all_docs['metadatas'][i]
                }
                export_data['documents'].append(doc_data)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            console.print(f"[green]成功导出 {count} 个文档到 {output_file}[/green]")
            
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")


# CLI 命令
@click.group()
@click.option('--index-dir', default='.codebase_index', help='索引目录')
@click.pass_context
def debug_cli(ctx, index_dir):
    """Debug utilities for codebase index"""
    ctx.obj = IndexDebugger(index_dir=index_dir)


@debug_cli.command()
@click.pass_obj
def inspect(debugger):
    """检查集合详细信息"""
    debugger.inspect_collection()


@debug_cli.command()
@click.option('--limit', '-l', default=5, help='显示文档数量')
@click.pass_obj
def samples(debugger, limit):
    """显示样本文档"""
    debugger.show_sample_documents(limit)


@debug_cli.command()
@click.pass_obj
def distribution(debugger):
    """分析文件分布"""
    debugger.analyze_file_distribution()


@debug_cli.command()
@click.argument('query')
@click.option('--results', '-n', default=5, help='搜索结果数量')
@click.pass_obj
def search_debug(debugger, query, results):
    """调试搜索功能"""
    debugger.search_debug(query, results)


@debug_cli.command()
@click.pass_obj
def hashes(debugger):
    """显示文件哈希缓存"""
    debugger.show_file_hashes()


@debug_cli.command()
@click.pass_obj
def validate(debugger):
    """验证索引完整性"""
    debugger.validate_index_integrity()


@debug_cli.command()
@click.option('--output', '-o', default='index_export.json', help='输出文件名')
@click.pass_obj
def export(debugger, output):
    """导出索引数据"""
    debugger.export_index_data(output)


@debug_cli.command()
@click.pass_obj
def all(debugger):
    """运行所有检查"""
    console.print("[bold yellow]运行所有检查...[/bold yellow]\n")
    
    debugger.inspect_collection()
    console.print("\n" + "="*50 + "\n")
    
    debugger.analyze_file_distribution()
    console.print("\n" + "="*50 + "\n")
    
    debugger.show_file_hashes()
    console.print("\n" + "="*50 + "\n")
    
    debugger.validate_index_integrity()
    console.print("\n" + "="*50 + "\n")
    
    debugger.show_sample_documents(3)


if __name__ == '__main__':
    debug_cli()