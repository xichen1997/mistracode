#!/usr/bin/env python3
"""
Test RAG decision functionality
"""

from codebase_agent_rag import CodebaseAgentRAG

def test_rag_decisions():
    """Test RAG decisions for various queries"""
    agent = CodebaseAgentRAG()
    
    # Test cases
    test_queries = [
        # Queries that should use RAG
        ("What does this function do?", True),
        ("What errors are in the code?", True),
        ("How to implement this feature?", True),
        ("Find all Python files", True),
        ("Explain the structure of this class", True),
        ("What modules are in the codebase?", True),
        ("Where is this variable defined?", True),
        ("How to fix this bug?", True),
        ("What is the purpose of this file?", True),
        ("What dependencies are used in the project?", True),
        
        # Queries that should be answered directly
        ("Hello", False),
        ("How's the weather today?", False),
        ("Thank you", False),
        ("Goodbye", False),
        ("What is artificial intelligence?", False),
        ("How to learn programming?", False),
        ("What is Python?", False),
        ("How are you?", False),
        ("What is machine learning?", False),
        ("How to improve programming skills?", False),
        ("What is object-oriented programming?", False),
        ("What is a database?", False),
    ]
    
    print("Testing RAG decision functionality\n")
    print("=" * 50)
    
    correct = 0
    total = len(test_queries)
    
    for query, expected in test_queries:
        result = agent.should_use_rag(query)
        status = "✅" if result == expected else "❌"
        decision = "RAG" if result else "DIRECT"
        expected_decision = "RAG" if expected else "DIRECT"
        
        print(f"{status} Query: {query}")
        print(f"   Decision: {decision} (Expected: {expected_decision})")
        print()
        
        if result == expected:
            correct += 1
    
    print("=" * 50)
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

if __name__ == "__main__":
    test_rag_decisions() 