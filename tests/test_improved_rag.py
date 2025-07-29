#!/usr/bin/env python3
"""
Test improved RAG judgment functionality
"""

from codebase_agent_rag import CodebaseAgentRAG
import time

def test_improved_rag():
    """Test improved RAG judgment functionality"""
    agent = CodebaseAgentRAG()
    
    # Comprehensive test cases
    test_queries = [
        # Obvious code-related questions (quick check)
        ("What is the function of this function?", True, "Quick check - obvious code-related"),
        ("Where is the bug in the code?", True, "Quick check - obvious code-related"),
        ("How is this class implemented?", True, "Quick check - obvious code-related"),
        ("What does this error mean?", True, "Quick check - obvious code-related"),
        ("Find all Python files", True, "Quick check - obvious code-related"),
        
        # Ambiguous queries requiring LLM judgment
        ("How to implement this feature?", True, "LLM judgment - asking about current project feature"),
        ("How to run this program?", True, "LLM judgment - asking about current project"),
        ("How does this project work?", True, "LLM judgment - asking about specific project"),
        ("How to optimize this code?", True, "LLM judgment - asking about current code"),
        ("How to fix this issue?", True, "LLM judgment - asking about current issue"),
        
        # General knowledge questions (LLM judgment)
        ("What is Python?", False, "LLM judgment - general knowledge"),
        ("How to learn programming?", False, "LLM judgment - general advice"),
        ("What is machine learning?", False, "LLM judgment - general concept"),
        ("How to improve code quality?", False, "LLM judgment - general advice"),
        ("How to debug code?", False, "LLM judgment - general skill"),
        ("What is object-oriented programming?", False, "LLM judgment - general concept"),
        ("What is a database?", False, "LLM judgment - general concept"),
        ("How to write tests?", False, "LLM judgment - general skill"),
        ("What are design patterns?", False, "LLM judgment - general concept"),
        
        # Greetings (fast check)
        ("Hello", False, "Fast check - greeting"),
        ("Thank you", False, "Fast check - greeting"),
        ("Goodbye", False, "Fast check - greeting"),
        ("How are you?", False, "Fast check - greeting"),
        
        # Boundary cases
        ("This", False, "Fast check - too short"),
        ("hi", False, "Fast check - greeting"),
        ("hello", False, "Fast check - greeting"),
    ]
    
    print("Testing improved RAG judgment functionality\n")
    print("=" * 70)
    
    correct = 0
    total = len(test_queries)
    llm_calls = 0
    fast_checks = 0
    
    for query, expected, reason in test_queries:
        print(f"\nQuery: {query}")
        print(f"Expected: {'RAG' if expected else 'DIRECT'} - {reason}")
        
        start_time = time.time()
        result = agent.should_use_rag(query)
        end_time = time.time()
        
        status = "✅" if result == expected else "❌"
        decision = "RAG" if result else "DIRECT"
        
        print(f"{status} Result: {decision} (Time: {end_time - start_time:.2f}s)")
        
        # Count LLM calls
        if "LLM judgment" in reason:
            llm_calls += 1
        else:
            fast_checks += 1
        
        if result == expected:
            correct += 1
        else:
            print(f"   ❌ Expected: {'RAG' if expected else 'DIRECT'}")
    
    print("\n" + "=" * 70)
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"Fast checks: {fast_checks} times")
    print(f"LLM calls: {llm_calls} times")
    
    if correct == total:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total - correct} tests failed")

def test_specific_cases():
    """Test specific boundary cases"""
    agent = CodebaseAgentRAG()
    
    print("\nTesting specific boundary cases")
    print("=" * 40)
    
    # Test previously failed cases
    failed_cases = [
        ("How to implement this feature?", True),
        ("How to run this program?", True),
        ("How to debug code?", False),
    ]
    
    for query, expected in failed_cases:
        print(f"\nTest: {query}")
        print(f"Expected: {'RAG' if expected else 'DIRECT'}")
        
        result = agent.should_use_rag(query)
        status = "✅" if result == expected else "❌"
        
        print(f"{status} Result: {'RAG' if result else 'DIRECT'}")

if __name__ == "__main__":
    test_improved_rag()
    test_specific_cases() 