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
        """check ChromaDB collection details"""
        console.print("[bold cyan]ChromaDB collection details[/bold cyan]")
        
        try:
            collection = self.agent.rag.collection
            count = collection.count()
            
            console.print(f"collection name: {collection.name}")
            console.print(f"total documents: {count}")
            
            if count > 0:
                # get all documents
                all_docs = collection.get(limit=count)
                
                console.print(f"IDs count: {len(all_docs['ids'])}")
                console.print(f"Documents count: {len(all_docs['documents'])}")
                console.print(f"Metadatas count: {len(all_docs['metadatas'])}")
                
                # analyze metadata fields
                if all_docs['metadatas']:
                    metadata_fields = set()
                    for metadata in all_docs['metadatas']:
                        metadata_fields.update(metadata.keys())
                    
                    console.print(f"metadata fields: {sorted(metadata_fields)}")
                
        except Exception as e:
            console.print(f"[red]error: {e}[/red]")
    
    def show_sample_documents(self, limit: int = 5):
        """show sample documents"""
        console.print(f"[bold cyan]sample documents (first {limit} documents)[/bold cyan]")
        
        try:
            collection = self.agent.rag.collection
            count = collection.count()
            
            if count == 0:
                console.print("[yellow]no documents[/yellow]")
                return
            
            sample_docs = collection.get(limit=min(limit, count))
            
            for i in range(len(sample_docs['ids'])):
                doc_id = sample_docs['ids'][i]
                content = sample_docs['documents'][i]
                metadata = sample_docs['metadatas'][i]
                
                console.print(f"\n[bold green]document {i+1}:[/bold green]")
                console.print(f"ID: {doc_id}")
                
                # show metadata
                metadata_table = Table(title="metadata")
                metadata_table.add_column("field", style="cyan")
                metadata_table.add_column("value", style="yellow")
                
                for key, value in metadata.items():
                    metadata_table.add_row(key, str(value))
                
                console.print(metadata_table)
                
                # show content preview
                preview = content[:200] + "..." if len(content) > 200 else content
                syntax = Syntax(preview, "python", theme="monokai", line_numbers=False)
                console.print(Panel(syntax, title="content preview", border_style="dim"))
        
        except Exception as e:
            console.print(f"[red]error: {e}[/red]")
    
    def analyze_file_distribution(self):
        """analyze file distribution"""
        console.print("[bold cyan]file distribution analysis[/bold cyan]")
        
        try:
            collection = self.agent.rag.collection
            count = collection.count()
            
            if count == 0:
                console.print("[yellow]no documents[/yellow]")
                return
            
            all_docs = collection.get(limit=count)
            
            # count file distribution
            file_chunks = {}
            chunk_types = {}
            
            for metadata in all_docs['metadatas']:
                file_path = metadata.get('file_path', 'unknown')
                chunk_type = metadata.get('chunk_type', 'unknown')
                
                file_chunks[file_path] = file_chunks.get(file_path, 0) + 1
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            # show file distribution
            file_table = Table(title="file code chunk distribution")
            file_table.add_column("file path", style="cyan")
            file_table.add_column("code chunk count", justify="right", style="green")
            
            for file_path, count in sorted(file_chunks.items(), key=lambda x: x[1], reverse=True):
                file_table.add_row(file_path, str(count))
            
            console.print(file_table)
            
            # show code chunk type distribution
            type_table = Table(title="code chunk type distribution")
            type_table.add_column("type", style="cyan")
            type_table.add_column("count", justify="right", style="green")
            
            for chunk_type, count in sorted(chunk_types.items(), key=lambda x: x[1], reverse=True):
                type_table.add_row(chunk_type, str(count))
            
            console.print(type_table)
            
        except Exception as e:
            console.print(f"[red]error: {e}[/red]")
    
    def search_debug(self, query: str, n_results: int = 5):
        """debug search functionality"""
        console.print(f"[bold cyan]search debug: {query}[/bold cyan]")
        
        try:
            # execute search
            results = self.agent.rag.search(query, n_results=n_results)
            
            if not results:
                console.print("[yellow]no search results[/yellow]")
                return
            
            console.print(f"found {len(results)} results")
            
            for i, result in enumerate(results):
                console.print(f"\n[bold green]result {i+1}:[/bold green]")
                console.print(f"ID: {result['id']}")
                console.print(f"similarity score: {1 - result['distance']:.4f}")
                console.print(f"distance: {result['distance']:.4f}")
                
                # show metadata
                metadata = result['metadata']
                console.print(f"file: {metadata.get('file_path', 'N/A')}")
                console.print(f"line number: {metadata.get('start_line', 'N/A')}-{metadata.get('end_line', 'N/A')}")
                console.print(f"type: {metadata.get('chunk_type', 'N/A')}")
                if metadata.get('name'):
                    console.print(f"name: {metadata['name']}")
                
                # show content preview
                content = result['content']
                preview = content[:150] + "..." if len(content) > 150 else content
                syntax = Syntax(preview, "python", theme="monokai", line_numbers=False)
                console.print(Panel(syntax, title="content", border_style="dim"))
        
        except Exception as e:
            console.print(f"[red]error: {e}[/red]")
    
    def show_file_hashes(self):
        """show file hash cache"""
        console.print("[bold cyan]file hash cache[/bold cyan]")
        
        hash_file = self.agent.rag.hash_cache_file
        
        if not hash_file.exists():
            console.print("[yellow]file hash cache file not found[/yellow]")
            return
        
        try:
            with open(hash_file, 'r') as f:
                hashes = json.load(f)
            
            if not hashes:
                console.print("[yellow]file hash cache is empty[/yellow]")
                return
            
            hash_table = Table(title="file hash")
            hash_table.add_column("file path", style="cyan")
            hash_table.add_column("MD5 hash", style="yellow")
            
            for file_path, file_hash in sorted(hashes.items()):
                hash_table.add_row(file_path, file_hash)
            
            console.print(hash_table)
            console.print(f"\ntotal files: {len(hashes)}")
            
        except Exception as e:
            console.print(f"[red]error: {e}[/red]")
    
    def validate_index_integrity(self):
        """validate index integrity"""
        console.print("[bold cyan]validate index integrity[/bold cyan]")
        
        try:
            collection = self.agent.rag.collection
            count = collection.count()
            
            if count == 0:
                console.print("[yellow]index is empty[/yellow]")
                return
            
            all_docs = collection.get(limit=count)
            
            issues = []
            
            # check required fields
            required_fields = ['file_path', 'start_line', 'end_line', 'chunk_type']
            
            for i, metadata in enumerate(all_docs['metadatas']):
                doc_id = all_docs['ids'][i]
                
                for field in required_fields:
                    if field not in metadata:
                        issues.append(f"document {doc_id}: missing field '{field}'")
                
                # check line number is reasonable
                start_line = metadata.get('start_line')
                end_line = metadata.get('end_line')
                
                if start_line and end_line:
                    if start_line > end_line:
                        issues.append(f"document {doc_id}: start line {start_line} > end line {end_line}")
                    if start_line < 1:
                        issues.append(f"document {doc_id}: start line {start_line} < 1")
            
            if issues:
                console.print(f"[red]found {len(issues)} issues:[/red]")
                for issue in issues:
                    console.print(f"  • {issue}")
            else:
                console.print("[green]index integrity check passed[/green]")
                
        except Exception as e:
            console.print(f"[red]error: {e}[/red]")
    
    def export_index_data(self, output_file: str = "index_export.json"):
        """export index data to JSON file"""
        console.print(f"[bold cyan]export index data to {output_file}[/bold cyan]")
        
        try:
            collection = self.agent.rag.collection
            count = collection.count()
            
            if count == 0:
                console.print("[yellow]no data to export[/yellow]")
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
            
            console.print(f"[green]successfully exported {count} documents to {output_file}[/green]")
            
        except Exception as e:
            console.print(f"[red]error: {e}[/red]")


# CLI commands
@click.group()
@click.option('--index-dir', default='.codebase_index', help='index directory')
@click.pass_context
def debug_cli(ctx, index_dir):
    """Debug utilities for codebase index"""
    ctx.obj = IndexDebugger(index_dir=index_dir)


@debug_cli.command()
@click.pass_obj
def inspect(debugger):
    """check collection details"""
    debugger.inspect_collection()


@debug_cli.command()
@click.option('--limit', '-l', default=5, help='number of documents to show')
@click.pass_obj
def samples(debugger, limit):
    """show sample documents"""
    debugger.show_sample_documents(limit)


@debug_cli.command()
@click.pass_obj
def distribution(debugger):
    """analyze file distribution"""
    debugger.analyze_file_distribution()


@debug_cli.command()
@click.argument('query')
@click.option('--results', '-n', default=5, help='number of search results')
@click.pass_obj
def search_debug(debugger, query, results):
    """debug search functionality"""
    debugger.search_debug(query, results)


@debug_cli.command()
@click.pass_obj
def hashes(debugger):
    """show file hash cache"""
    debugger.show_file_hashes()


@debug_cli.command()
@click.pass_obj
def validate(debugger):
    """validate index integrity"""
    debugger.validate_index_integrity()


@debug_cli.command()
@click.option('--output', '-o', default='index_export.json', help='output file name')
@click.pass_obj
def export(debugger, output):
    """export index data"""
    debugger.export_index_data(output)


@debug_cli.command()
@click.pass_obj
def all(debugger):
    """run all checks"""
    console.print("[bold yellow]running all checks...[/bold yellow]\n")
    
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