.PHONY: help run-coordinator run-detector run-workflow run-all run-database clean venv

# Python environment
PYTHON := python3
VENV := venv
ifeq ($(OS),Windows_NT)
	VENV_ACTIVATE := $(VENV)\Scripts\activate
else
	VENV_ACTIVATE := . $(VENV)/bin/activate &&
endif

help:
	@echo "Code Smell Detection System - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make venv              - Create Python virtual environment"
	@echo "  make install           - Install dependencies"
	@echo ""
	@echo "Execution:"
	@echo "  make run-detector      - Run Deep Agent detector"
	@echo "  make run-workflow      - Run LangGraph workflow pipeline"
	@echo "  make run-coordinator   - Run Analysis Coordinator"
	@echo "  make run-database      - Run database manager validation"
	@echo "  make run-all           - Run complete integrated system"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean             - Remove cache and temporary files"

venv:
	$(PYTHON) -m venv $(VENV)
	@echo "✓ Virtual environment created. Activate with: source $(VENV)/bin/activate"

install:
	$(PYTHON) -m pip install -r requirements.txt
	@echo "✓ Dependencies installed"

run-detector:
	$(VENV_ACTIVATE) $(PYTHON) -c "from src.code_smell_detector import CodeSmellDetector; d = CodeSmellDetector(specialization='Long Method expert'); print('✓ Deep Agent detector initialized'); s = d.get_stats(); print(f'Framework: {s[\"framework\"]}'); print(f'Tools available: {len(s[\"tools_available\"])}')"

run-workflow:
	$(VENV_ACTIVATE) $(PYTHON) -c "from src.workflow_graph import build_workflow_graph; g = build_workflow_graph(); print('✓ LangGraph workflow compiled'); print('✓ Ready for analysis')"

run-coordinator:
	$(VENV_ACTIVATE) $(PYTHON) -c "from src.analysis_coordinator import AnalysisCoordinator; c = AnalysisCoordinator(); print('✓ Analysis Coordinator initialized'); print(f'Detectors: {len(c.detectors)}')"

run-database:
	@echo "Testing Database Manager..."
	@$(VENV_ACTIVATE) $(PYTHON) -c "from src.database_manager import DatabaseManager; db = DatabaseManager(); stats = db.get_database_stats(); print('✓ Database manager initialized'); print(f'Tables: {len(stats) - 1}'); print(f'Database size: {stats[\"database_size_mb\"]:.2f} MB')"
	@echo "✓ Database manager operational"

run-all:
	@echo "Running Complete System Verification..."
	@echo ""
	@echo "1. Initializing Deep Agent Detector..."
	@$(VENV_ACTIVATE) $(PYTHON) -c "from src.code_smell_detector import CodeSmellDetector; d = CodeSmellDetector(); print('✓ Detector OK')"
	@echo ""
	@echo "2. Building LangGraph Workflow..."
	@$(VENV_ACTIVATE) $(PYTHON) -c "from src.workflow_graph import build_workflow_graph; g = build_workflow_graph(); print('✓ Workflow OK')"
	@echo ""
	@echo "3. Initializing Analysis Coordinator..."
	@$(VENV_ACTIVATE) $(PYTHON) -c "from src.analysis_coordinator import AnalysisCoordinator; c = AnalysisCoordinator(); print('✓ Coordinator OK')"
	@echo ""
	@echo "4. Testing Database Manager..."
	@$(VENV_ACTIVATE) $(PYTHON) -c "from src.database_manager import DatabaseManager; db = DatabaseManager(); stats = db.get_database_stats(); print('✓ Database OK')"
	@echo ""
	@echo "✅ All systems operational and ready for Phase 3"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
	@echo "✓ Cache cleaned"
