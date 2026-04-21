#!/usr/bin/env python3
"""
Test script for LangGraph + FastAPI integration with agentic model selection
"""

import asyncio
import json
from datetime import datetime

from src.workflow.workflow_graph import WorkflowExecutor
from src.llm.llm_client import OllamaClient


async def test_ollama_connection():
    """Test 1: Verify Ollama connection and available models"""
    print("\n" + "=" * 70)
    print("TEST 1: Ollama Connection & Available Models")
    print("=" * 70)

    try:
        client = OllamaClient()
        models = client.get_available_models()
        print(f"✅ Successfully fetched models from Ollama")
        print(f"   Available models: {models}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Ollama: {e}")
        print("   Make sure Ollama is running: ollama serve")
        return False


async def test_auto_model_selection():
    """Test 2: Agentic model selection (auto-select)"""
    print("\n" + "=" * 70)
    print("TEST 2: Agentic Model Selection (Auto-Select)")
    print("=" * 70)

    code_small = """def hello():
    print("Hello, world!")
    return 42
"""

    code_large = """def complex_algorithm():
    # 100+ line complex algorithm
    """ + "\n    ".join(f"x = {i}" for i in range(50))

    executor = WorkflowExecutor()

    print("\n--- Small Code (auto-select) ---")
    try:
        result_small = await executor.execute(
            code_snippet=code_small,
            file_name="small.py"
        )
        print(f"✅ Small code analysis complete")
        print(f"   Selected model: {result_small['metadata'].get('model_used', 'N/A')}")
        print(f"   Reasoning: {result_small['metadata'].get('model_reasoning', 'N/A')}")
    except Exception as e:
        print(f"❌ Small code analysis failed: {e}")
        return False

    print("\n--- Large Code (auto-select) ---")
    try:
        result_large = await executor.execute(
            code_snippet=code_large,
            file_name="large.py"
        )
        print(f"✅ Large code analysis complete")
        print(f"   Selected model: {result_large['metadata'].get('model_used', 'N/A')}")
        print(f"   Reasoning: {result_large['metadata'].get('model_reasoning', 'N/A')}")
    except Exception as e:
        print(f"❌ Large code analysis failed: {e}")
        return False

    return True


async def test_manual_model_selection():
    """Test 3: Manual model selection"""
    print("\n" + "=" * 70)
    print("TEST 3: Manual Model Selection")
    print("=" * 70)

    code = """def sample():
    pass
"""

    executor = WorkflowExecutor()
    models_to_test = ["llama3:8b", "mistral:7b", "codellama:13b"]

    for model in models_to_test:
        print(f"\n--- Testing with {model} ---")
        try:
            result = await executor.execute(
                code_snippet=code,
                file_name="sample.py",
                model=model
            )
            print(f"✅ Analysis with {model} complete")
            print(f"   Used model: {result['metadata'].get('model_used', 'N/A')}")
        except Exception as e:
            print(f"❌ Analysis with {model} failed: {e}")
            return False

    return True


async def test_workflow_nodes():
    """Test 4: Verify all workflow nodes execute"""
    print("\n" + "=" * 70)
    print("TEST 4: Workflow Node Execution")
    print("=" * 70)

    code = """def process():
    # Some code here
    data = [1, 2, 3]
    result = sum(data)
    return result
"""

    executor = WorkflowExecutor()

    try:
        result = await executor.execute(
            code_snippet=code,
            file_name="process.py"
        )

        print(f"✅ Workflow executed successfully")
        print(f"\n   Workflow Results:")
        print(f"   - Detections: {len(result.get('detections', []))} found")
        print(f"   - Validated findings: {len(result.get('validated_findings', []))} confirmed")
        print(f"   - Errors: {len(result.get('errors', []))} errors")
        print(f"   - Language detected: {result.get('language', 'unknown')}")
        print(f"   - Model used: {result['metadata'].get('model_used', 'N/A')}")

        if result.get('errors'):
            print(f"\n   ⚠️  Errors encountered:")
            for error in result.get('errors', []):
                print(f"      - {error}")

        return True
    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_state_fields():
    """Test 5: Verify AnalysisState contains all new fields"""
    print("\n" + "=" * 70)
    print("TEST 5: AnalysisState New Fields")
    print("=" * 70)

    from src.workflow.workflow_graph import AnalysisState

    try:
        initial_state = AnalysisState(
            code_snippet="def test(): pass"
        )

        required_fields = ['model', 'use_rag', 'available_models', 'model_reasoning']
        missing_fields = []

        for field in required_fields:
            if not hasattr(initial_state, field):
                missing_fields.append(field)

        if missing_fields:
            print(f"❌ Missing fields in AnalysisState: {missing_fields}")
            return False

        print(f"✅ All required fields present in AnalysisState:")
        for field in required_fields:
            print(f"   - {field}: {getattr(initial_state, field, 'N/A')}")

        return True
    except Exception as e:
        print(f"❌ Failed to verify AnalysisState: {e}")
        return False


async def test_api_model_selection():
    """Test 6: Verify API model selection parameter"""
    print("\n" + "=" * 70)
    print("TEST 6: API Model Selection Parameter")
    print("=" * 70)

    from src.api.models import CodeSubmissionRequest

    try:
        # Test request without model
        req1 = CodeSubmissionRequest(
            code="def foo(): pass",
            file_name="test.py"
        )
        print(f"✅ Request without model: model={req1.model}")

        # Test request with model
        req2 = CodeSubmissionRequest(
            code="def foo(): pass",
            file_name="test.py",
            model="llama3:8b"
        )
        print(f"✅ Request with model: model={req2.model}")

        return True
    except Exception as e:
        print(f"❌ Failed to verify API model parameter: {e}")
        return False


async def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("LangGraph + FastAPI Integration Test Suite")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")

    tests = [
        ("Ollama Connection", test_ollama_connection),
        ("API Model Parameter", test_api_model_selection),
        ("AnalysisState Fields", test_state_fields),
        ("Auto Model Selection", test_auto_model_selection),
        ("Manual Model Selection", test_manual_model_selection),
        ("Workflow Execution", test_workflow_nodes),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = await test_func()
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print(f"Completed: {datetime.now().isoformat()}")

    if passed == total:
        print("\n🎉 All tests passed! Integration is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. See details above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
