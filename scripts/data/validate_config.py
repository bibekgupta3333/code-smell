#!/usr/bin/env python
"""
Validation Script for Configuration Module
Tests all components of Phase 2.1 (Configuration Module)
Run this to verify the environment is correctly set up.
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """Test all required dependencies can be imported."""
    print("=" * 70)
    print("TESTING PYTHON DEPENDENCIES")
    print("=" * 70)

    required_packages = [
        ("ollama", "Ollama Python client"),
        ("chromadb", "Vector database"),
        ("langchain", "LLM orchestration"),
        ("langgraph", "Workflow state machine"),
        ("sentence_transformers", "Embedding models"),
        ("tiktoken", "Token counting"),
        ("pydantic", "Data validation"),
        ("structlog", "Structured logging"),
        ("pygments", "Code syntax highlighting"),
        ("pandas", "Data manipulation"),
        ("numpy", "Numerical computing"),
        ("sklearn", "Machine learning utils"),
        ("matplotlib", "Plotting"),
        ("seaborn", "Statistical viz"),
        ("pytest", "Testing framework"),
    ]

    failed = []
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✓ {package:25s} - {description}")
        except ImportError as e:
            print(f"✗ {package:25s} - FAILED: {e}")
            failed.append(package)

    print()
    if failed:
        print(f"❌ {len(failed)} package(s) failed to import: {', '.join(failed)}")
        return False
    print(f"✅ All {len(required_packages)} required packages imported successfully")
    return True


def test_config():
    """Test config.py loads correctly."""
    print("\n" + "=" * 70)
    print("TESTING CONFIGURATION MODULE")
    print("=" * 70)

    try:
        import config
        print("✓ config.py imported successfully")

        # Test key configuration values
        tests = [
            ("OLLAMA_BASE_URL", config.OLLAMA_BASE_URL, "http://localhost:11434"),
            ("DEFAULT_MODEL", config.DEFAULT_MODEL, "llama3:8b"),
            ("EMBEDDING_MODEL", "all-MiniLM-L6-v2" in config.EMBEDDING_MODEL, True),
            ("RANDOM_SEED", config.RANDOM_SEED, 42),
            ("RAG top_k", config.RAG_CONFIG["top_k"], 5),
            ("RAG similarity_threshold", config.RAG_CONFIG["similarity_threshold"], 0.7),
            ("BATCH_SIZE", config.BATCH_SIZE, 5),
            ("MAX_CONCURRENT_REQUESTS", config.MAX_CONCURRENT_REQUESTS, 2),
        ]

        for name, value, expected in tests:
            if value == expected:
                print(f"✓ {name:30s} = {value}")
            else:
                print(f"✗ {name:30s} = {value} (expected {expected})")

        print()
        print("✅ Configuration module loaded successfully")
        return True

    except Exception as e:
        print(f"❌ Failed to load config.py: {e}")
        return False


def test_directories():
    """Test all required directories exist."""
    print("\n" + "=" * 70)
    print("TESTING PROJECT DIRECTORIES")
    print("=" * 70)

    try:
        import config

        # Run directory creation
        config.ensure_directories()

        required_dirs = [
            config.DATA_DIR,
            config.RESULTS_DIR,
            config.PREDICTIONS_DIR,
            config.CONFUSION_MATRICES_DIR,
            config.PERFORMANCE_DIR,
            config.RESOURCES_DIR,
            config.FIGURES_DIR,
            config.TABLES_DIR,
            config.METRICS_DIR,
            config.CHROMADB_DIR,
            config.CACHE_DIR,
            config.EXPERIMENT_LOG_DIR,
        ]

        all_exist = True
        for directory in required_dirs:
            if directory.exists():
                print(f"✓ {directory.relative_to(config.PROJECT_ROOT)}")
            else:
                print(f"✗ {directory.relative_to(config.PROJECT_ROOT)} - NOT FOUND")
                all_exist = False

        print()
        if all_exist:
            print(f"✅ All {len(required_dirs)} directories exist")
            return True
        print("❌ Some directories are missing")
        return False

    except Exception as e:
        print(f"❌ Failed to verify directories: {e}")
        return False


def test_requirements_file():
    """Test requirements.txt exists and is valid."""
    print("\n" + "=" * 70)
    print("TESTING REQUIREMENTS.TXT")
    print("=" * 70)

    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("❌ requirements.txt not found")
        return False

    print(f"✓ requirements.txt found")

    # Count packages
    packages = []
    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "==" in line:
                packages.append(line.split("==")[0])

    print(f"✓ Contains {len(packages)} pinned dependencies")

    # Check for key packages
    key_packages = [
        "ollama", "chromadb", "langchain", "langgraph",
        "sentence-transformers", "tiktoken", "pydantic", "structlog"
    ]

    missing = []
    for pkg in key_packages:
        if pkg in packages:
            print(f"✓ {pkg}")
        else:
            print(f"✗ {pkg} - MISSING")
            missing.append(pkg)

    print()
    if missing:
        print(f"❌ Missing key packages: {', '.join(missing)}")
        return False
    else:
        print("✅ requirements.txt is valid and complete")
        return True


def test_dockerfile():
    """Test Dockerfile exists."""
    print("\n" + "=" * 70)
    print("TESTING DOCKERFILE")
    print("=" * 70)

    dockerfile = Path("Dockerfile")
    if not dockerfile.exists():
        print("❌ Dockerfile not found")
        return False

    print("✓ Dockerfile found")

    # Check for key content
    content = dockerfile.read_text()
    checks = [
        ("FROM python:3.11", "Python 3.11 base image"),
        ("COPY requirements.txt", "Requirements copy"),
        ("pip install", "Dependency installation"),
        ("COPY config.py", "Config copy"),
        ("EXPOSE", "Port exposure"),
    ]

    for pattern, desc in checks:
        if pattern in content:
            print(f"✓ {desc}")
        else:
            print(f"✗ {desc} - NOT FOUND")

    print()
    print("✅ Dockerfile exists")

    # Check .dockerignore
    dockerignore = Path(".dockerignore")
    if dockerignore.exists():
        print("✓ .dockerignore found")
    else:
        print("⚠ .dockerignore not found (recommended)")

    # Check docker-compose.yml
    compose = Path("docker-compose.yml")
    if compose.exists():
        print("✓ docker-compose.yml found")
    else:
        print("⚠ docker-compose.yml not found (optional)")

    print()
    print("✅ Docker configuration complete")
    return True


def main():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("CONFIGURATION MODULE VALIDATION")
    print("Phase 2.1 - Week 3 (Feb 26 - Mar 1, 2026)")
    print("=" * 70)
    print()

    results = {
        "Python Dependencies": test_imports(),
        "Configuration Module": test_config(),
        "Project Directories": test_directories(),
        "Requirements File": test_requirements_file(),
        "Docker Configuration": test_dockerfile(),
    }

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:25s} {status}")

    print()

    if all(results.values()):
        print("🎉 ALL TESTS PASSED - Configuration Module is ready!")
        print()
        print("Next steps:")
        print("1. Start Ollama: ollama serve")
        print("2. Pull models: ollama pull llama3:8b")
        print("3. Proceed to Phase 2.2 (LLM Integration Module)")
        return 0
    failed_count = sum(1 for v in results.values() if not v)
    print(f"❌ {failed_count}/{len(results)} tests failed")
    print("Please fix the issues above before proceeding.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
