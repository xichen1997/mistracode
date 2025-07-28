#!/usr/bin/env python3
"""
测试 LLM 判断功能
"""

from codebase_agent_rag import CodebaseAgentRAG
import time

def test_llm_judgment():
    """测试 LLM 判断功能"""
    agent = CodebaseAgentRAG()
    
    # 边界测试用例
    test_queries = [
        # 明显的代码相关问题
        ("这个函数的作用是什么？", True, "应该使用 RAG"),
        ("代码中的 bug 在哪里？", True, "应该使用 RAG"),
        ("这个类是如何实现的？", True, "应该使用 RAG"),
        ("这个错误是什么意思？", True, "询问具体错误，应该使用 RAG"),
        
        # 模糊的查询 - 需要 LLM 判断
        ("什么是 Python？", False, "通用知识，应该直接回答"),
        ("如何学习编程？", False, "通用建议，应该直接回答"),
        ("什么是机器学习？", False, "通用概念，应该直接回答"),
        ("如何提高代码质量？", False, "通用建议，应该直接回答"),
        ("如何调试代码？", False, "通用技能，应该直接回答"),
        
        # 需要 LLM 判断的模糊查询
        ("如何实现这个功能？", True, "询问当前项目功能，应该使用 RAG"),
        ("如何运行这个程序？", True, "询问当前项目，应该使用 RAG"),
        ("这个项目是如何工作的？", True, "询问具体项目，应该使用 RAG"),
        ("如何优化这个代码？", True, "询问当前代码，应该使用 RAG"),
        
        # 通用问题
        ("什么是面向对象编程？", False, "通用概念，应该直接回答"),
        ("什么是数据库？", False, "通用概念，应该直接回答"),
        ("如何编写测试？", False, "通用技能，应该直接回答"),
        ("什么是设计模式？", False, "通用概念，应该直接回答"),
    ]
    
    print("测试 LLM 判断功能\n")
    print("=" * 60)
    
    correct = 0
    total = len(test_queries)
    
    for query, expected, reason in test_queries:
        print(f"\n查询: {query}")
        print(f"期望: {'RAG' if expected else 'DIRECT'} - {reason}")
        
        start_time = time.time()
        result = agent.should_use_rag(query)
        end_time = time.time()
        
        status = "✅" if result == expected else "❌"
        decision = "RAG" if result else "DIRECT"
        
        print(f"{status} 结果: {decision} (耗时: {end_time - start_time:.2f}s)")
        
        if result == expected:
            correct += 1
        else:
            print(f"   ❌ 期望: {'RAG' if expected else 'DIRECT'}")
    
    print("\n" + "=" * 60)
    print(f"正确率: {correct}/{total} ({correct/total*100:.1f}%)")
    
    if correct == total:
        print("🎉 所有测试通过！")
    else:
        print(f"⚠️  有 {total - correct} 个测试失败")

def interactive_test():
    """交互式测试"""
    agent = CodebaseAgentRAG()
    
    print("交互式 LLM 判断测试")
    print("输入 'quit' 退出")
    print("=" * 40)
    
    while True:
        query = input("\n请输入查询: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        print(f"\n查询: {query}")
        print("正在分析...")
        
        start_time = time.time()
        result = agent.should_use_rag(query)
        end_time = time.time()
        
        decision = "🔍 RAG (需要搜索代码库)" if result else "💬 DIRECT (直接回答)"
        print(f"结果: {decision}")
        print(f"耗时: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_test()
    else:
        test_llm_judgment() 