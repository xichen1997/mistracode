#!/usr/bin/env python3
"""
Test LLM judgment functionality
"""

from codebase_agent_rag import CodebaseAgentRAG
import time

def test_llm_judgment():
    """Test LLM judgment functionality"""
    agent = CodebaseAgentRAG()
    
    # Boundary test cases
    test_queries = [
        # Obvious code-related questions
        ("What is the function of this function?", True, "Should use RAG"),
        ("Where is the bug in the code?", True, "Should use RAG"),
        ("How is this class implemented?", True, "Should use RAG"),
        ("What does this error mean?", True, "Asking about specific error, should use RAG"),
        
        # Ambiguous queries - need LLM judgment
        ("What is Python?", False, "General knowledge, should answer directly"),
        ("How to learn programming?", False, "General advice, should answer directly"),
        ("What is machine learning?", False, "General concept, should answer directly"),
        ("How to improve code quality?", False, "General advice, should answer directly"),
        ("How to debug code?", False, "General skill, should answer directly"),
        
        # Ambiguous queries that need LLM judgment
        ("How to implement this feature?", True, "Asking about current project feature, should use RAG"),
        ("How to run this program?", True, "Asking about current project, should use RAG"),
        ("How does this project work?", True, "Asking about specific project, should use RAG"),
        ("How to optimize this code?", True, "Asking about current code, should use RAG"),
        
        # General questions
        ("What is object-oriented programming?", False, "General concept, should answer directly"),
        ("What is a database?", False, "General concept, should answer directly"),
        ("How to write tests?", False, "General skill, should answer directly"),
        ("What are design patterns?", False, "General concept, should answer directly"),
    ]
    
    print("Testing LLM judgment functionality\n")
    print("=" * 60)
    
    correct = 0
    total = len(test_queries)
    
    for query, expected, reason in test_queries:
        print(f"\nQuery: {query}")
        print(f"Expected: {'RAG' if expected else 'DIRECT'} - {reason}")
        
        start_time = time.time()
        result = agent.should_use_rag(query)
        end_time = time.time()
        
        status = "✅" if result == expected else "❌"
        decision = "RAG" if result else "DIRECT"
        
        print(f"{status} Result: {decision} (Time: {end_time - start_time:.2f}s)")
        
        if result == expected:
            correct += 1
        else:
            print(f"   ❌ Expected: {'RAG' if expected else 'DIRECT'}")
    
    print("\n" + "=" * 60)
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    
    if correct == total:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total - correct} tests failed")

def interactive_test():
    """Interactive test"""
    agent = CodebaseAgentRAG()
    
    print("Interactive LLM judgment test")
    print("Enter 'quit' to exit")
    print("=" * 40)
    
    while True:
        query = input("\nPlease enter query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        print(f"\nQuery: {query}")
        print("Analyzing...")
        
        start_time = time.time()
        result = agent.should_use_rag(query)
        end_time = time.time()
        
        decision = "🔍 RAG (need to search codebase)" if result else "💬 DIRECT (direct answer)"
        print(f"Result: {decision}")
        print(f"Time: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_test()
    else:
        test_llm_judgment() 