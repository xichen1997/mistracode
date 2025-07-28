#!/usr/bin/env python3
"""
测试 RAG 决策功能
"""

from codebase_agent_rag import CodebaseAgentRAG

def test_rag_decisions():
    """测试各种查询的 RAG 决策"""
    agent = CodebaseAgentRAG()
    
    # 测试用例
    test_queries = [
        # 应该使用 RAG 的查询
        ("这个函数是做什么的？", True),
        ("代码中有哪些错误？", True),
        ("如何实现这个功能？", True),
        ("查找所有 Python 文件", True),
        ("解释这个类的结构", True),
        ("代码库中有哪些模块？", True),
        ("这个变量在哪里定义的？", True),
        ("如何修复这个 bug？", True),
        ("这个文件的作用是什么？", True),
        ("项目中使用了哪些依赖？", True),
        
        # 应该直接回答的查询
        ("你好", False),
        ("今天天气怎么样？", False),
        ("谢谢", False),
        ("再见", False),
        ("什么是人工智能？", False),
        ("如何学习编程？", False),
        ("Python 是什么？", False),
        ("你好吗？", False),
        ("什么是机器学习？", False),
        ("如何提高编程技能？", False),
        ("什么是面向对象编程？", False),
        ("数据库是什么？", False),
    ]
    
    print("测试 RAG 决策功能\n")
    print("=" * 50)
    
    correct = 0
    total = len(test_queries)
    
    for query, expected in test_queries:
        result = agent.should_use_rag(query)
        status = "✅" if result == expected else "❌"
        decision = "RAG" if result else "直接回答"
        expected_decision = "RAG" if expected else "直接回答"
        
        print(f"{status} 查询: {query}")
        print(f"   决策: {decision} (期望: {expected_decision})")
        print()
        
        if result == expected:
            correct += 1
    
    print("=" * 50)
    print(f"正确率: {correct}/{total} ({correct/total*100:.1f}%)")

if __name__ == "__main__":
    test_rag_decisions() 