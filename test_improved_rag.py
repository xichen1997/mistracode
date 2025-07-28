#!/usr/bin/env python3
"""
测试改进后的 RAG 判断功能
"""

from codebase_agent_rag import CodebaseAgentRAG
import time

def test_improved_rag():
    """测试改进后的 RAG 判断功能"""
    agent = CodebaseAgentRAG()
    
    # 全面的测试用例
    test_queries = [
        # 明显的代码相关问题（快速检查）
        ("这个函数的作用是什么？", True, "快速检查 - 明显代码相关"),
        ("代码中的 bug 在哪里？", True, "快速检查 - 明显代码相关"),
        ("这个类是如何实现的？", True, "快速检查 - 明显代码相关"),
        ("这个错误是什么意思？", True, "快速检查 - 明显代码相关"),
        ("查找所有 Python 文件", True, "快速检查 - 明显代码相关"),
        
        # 需要 LLM 判断的模糊查询
        ("如何实现这个功能？", True, "LLM 判断 - 询问当前项目功能"),
        ("如何运行这个程序？", True, "LLM 判断 - 询问当前项目"),
        ("这个项目是如何工作的？", True, "LLM 判断 - 询问具体项目"),
        ("如何优化这个代码？", True, "LLM 判断 - 询问当前代码"),
        ("如何修复这个问题？", True, "LLM 判断 - 询问当前问题"),
        
        # 通用知识问题（LLM 判断）
        ("什么是 Python？", False, "LLM 判断 - 通用知识"),
        ("如何学习编程？", False, "LLM 判断 - 通用建议"),
        ("什么是机器学习？", False, "LLM 判断 - 通用概念"),
        ("如何提高代码质量？", False, "LLM 判断 - 通用建议"),
        ("如何调试代码？", False, "LLM 判断 - 通用技能"),
        ("什么是面向对象编程？", False, "LLM 判断 - 通用概念"),
        ("什么是数据库？", False, "LLM 判断 - 通用概念"),
        ("如何编写测试？", False, "LLM 判断 - 通用技能"),
        ("什么是设计模式？", False, "LLM 判断 - 通用概念"),
        
        # 问候语（快速检查）
        ("你好", False, "快速检查 - 问候语"),
        ("谢谢", False, "快速检查 - 问候语"),
        ("再见", False, "快速检查 - 问候语"),
        ("你好吗？", False, "快速检查 - 问候语"),
        
        # 边界情况
        ("这个", False, "快速检查 - 太短"),
        ("hi", False, "快速检查 - 问候语"),
        ("hello", False, "快速检查 - 问候语"),
    ]
    
    print("测试改进后的 RAG 判断功能\n")
    print("=" * 70)
    
    correct = 0
    total = len(test_queries)
    llm_calls = 0
    fast_checks = 0
    
    for query, expected, reason in test_queries:
        print(f"\n查询: {query}")
        print(f"期望: {'RAG' if expected else 'DIRECT'} - {reason}")
        
        start_time = time.time()
        result = agent.should_use_rag(query)
        end_time = time.time()
        
        status = "✅" if result == expected else "❌"
        decision = "RAG" if result else "DIRECT"
        
        print(f"{status} 结果: {decision} (耗时: {end_time - start_time:.2f}s)")
        
        # 统计 LLM 调用次数
        if "LLM 判断" in reason:
            llm_calls += 1
        else:
            fast_checks += 1
        
        if result == expected:
            correct += 1
        else:
            print(f"   ❌ 期望: {'RAG' if expected else 'DIRECT'}")
    
    print("\n" + "=" * 70)
    print(f"正确率: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"快速检查: {fast_checks} 次")
    print(f"LLM 调用: {llm_calls} 次")
    
    if correct == total:
        print("🎉 所有测试通过！")
    else:
        print(f"⚠️  有 {total - correct} 个测试失败")

def test_specific_cases():
    """测试特定的边界情况"""
    agent = CodebaseAgentRAG()
    
    print("\n测试特定边界情况")
    print("=" * 40)
    
    # 测试之前失败的案例
    failed_cases = [
        ("如何实现这个功能？", True),
        ("如何运行这个程序？", True),
        ("如何调试代码？", False),
    ]
    
    for query, expected in failed_cases:
        print(f"\n测试: {query}")
        print(f"期望: {'RAG' if expected else 'DIRECT'}")
        
        result = agent.should_use_rag(query)
        status = "✅" if result == expected else "❌"
        
        print(f"{status} 结果: {'RAG' if result else 'DIRECT'}")

if __name__ == "__main__":
    test_improved_rag()
    test_specific_cases() 