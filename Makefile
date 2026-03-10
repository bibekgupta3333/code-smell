.PHONY: help venv install clean \
	run-detector run-workflow run-coordinator run-database run-all \
	baseline-install baseline-install-verify baseline-java baseline-python baseline-js baseline-run baseline-report baseline-verify \
	llm-baseline-run llm-rag-run llm-ablation-run

# Python environment
PYTHON := python3
VENV := venv
ifeq ($(OS),Windows_NT)
	VENV_ACTIVATE := $(VENV)\Scripts\activate
else
	VENV_ACTIVATE := . $(VENV)/bin/activate &&
endif

# Default directories
DIR ?= data/datasets/SmellyCodeDataset
TIMEOUT ?= 60

help:
	@echo "Code Smell Detection System - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make venv                - Create Python virtual environment"
	@echo "  make install             - Install Python dependencies"
	@echo ""
	@echo "System Verification:"
	@echo "  make run-detector        - Run Deep Agent detector"
	@echo "  make run-workflow        - Run LangGraph workflow pipeline"
	@echo "  make run-coordinator     - Run Analysis Coordinator"
	@echo "  make run-database        - Run database manager validation"
	@echo "  make run-all             - Run complete integrated system check"
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "  BASELINE TOOLS (Static Analysis)       scripts/baseline/"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""
	@echo "  make baseline-install        - Full setup: Docker, SonarQube, PMD, Checkstyle,"
	@echo "                                 SpotBugs, pylint, flake8, ESLint, results dirs"
	@echo "  make baseline-install-verify - Verify existing installation (no changes)"
	@echo ""
	@echo "  make baseline-java    DIR=<dir>  - Run Java tools (PMD, Checkstyle, SpotBugs, SonarQube)"
	@echo "  make baseline-python  DIR=<dir>  - Run Python tools (pylint, flake8)"
	@echo "  make baseline-js      DIR=<dir>  - Run JavaScript tools (ESLint)"
	@echo "  make baseline-run     DIR=<dir>  - Auto-detect language & run all tools"
	@echo ""
	@echo "  make baseline-report             - Generate summary tables & visualizations"
	@echo "  make baseline-verify             - Verify benchmarking modules (imports)"
	@echo ""
	@echo "  Options:  DIR=path  TIMEOUT=60  TOOLS=pmd,checkstyle  VERBOSE=1"
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "  LLM EXPERIMENTS                        scripts/"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""
	@echo "  make llm-baseline-run  [MODEL=llama3:8b] [DIR=data/datasets/marv/]"
	@echo "  make llm-rag-run       [MODEL=llama3:8b] [K=5] [DIR=data/datasets/marv/]"
	@echo "  make llm-ablation-run  [DIR=data/datasets/marv/]"
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
	$(VENV_ACTIVATE) $(PYTHON) -c "from src.analysis.code_smell_detector import CodeSmellDetector; d = CodeSmellDetector(specialization='Long Method expert'); print('✓ Deep Agent detector initialized'); s = d.get_stats(); print(f'Framework: {s.get(\"framework\")}'); print(f'Tools available: {len(s.get(\"tools_available\", []))}')"

run-workflow:
	$(VENV_ACTIVATE) $(PYTHON) -c "from src.workflow.workflow_graph import build_workflow_graph; g = build_workflow_graph(); print('✓ LangGraph workflow compiled'); print('✓ Ready for analysis')"

run-coordinator:
	$(VENV_ACTIVATE) $(PYTHON) -c "from src.workflow.analysis_coordinator import AnalysisCoordinator; c = AnalysisCoordinator(); print('✓ Analysis Coordinator initialized'); print(f'Detectors: {len(c.detectors)}')"

run-database:
	@echo "Testing Database Manager..."
	@$(VENV_ACTIVATE) $(PYTHON) -c "from src.database.database_manager import DatabaseManager; db = DatabaseManager(); stats = db.get_database_stats(); print('✓ Database manager initialized'); print(f'Tables: {len(stats) - 1}'); print(f'Database size: {stats.get(\"database_size_mb\", 0):.2f} MB')"
	@echo "✓ Database manager operational"

run-all:
	@echo "Running Complete System Verification..."
	@echo ""
	@echo "1. Initializing Deep Agent Detector..."
	@$(VENV_ACTIVATE) $(PYTHON) -c "from src.analysis.code_smell_detector import CodeSmellDetector; d = CodeSmellDetector(); print('✓ Detector OK')"
	@echo ""
	@echo "2. Building LangGraph Workflow..."
	@$(VENV_ACTIVATE) $(PYTHON) -c "from src.workflow.workflow_graph import build_workflow_graph; g = build_workflow_graph(); print('✓ Workflow OK')"
	@echo ""
	@echo "3. Initializing Analysis Coordinator..."
	@$(VENV_ACTIVATE) $(PYTHON) -c "from src.workflow.analysis_coordinator import AnalysisCoordinator; c = AnalysisCoordinator(); print('✓ Coordinator OK')"
	@echo ""
	@echo "4. Testing Database Manager..."
	@$(VENV_ACTIVATE) $(PYTHON) -c "from src.database.database_manager import DatabaseManager; db = DatabaseManager(); stats = db.get_database_stats(); print('✓ Database OK')"
	@echo ""
	@echo "✅ All systems operational and ready for Phase 3"

run-benchmark-setup:
	@echo "Setting up baseline tools..."
	@bash scripts/baseline/install_tools.sh

run-baseline-tools:
	@echo "Running baseline tools on test sample..."
	@$(VENV_ACTIVATE) $(PYTHON) scripts/baseline/run_tools.py --input tools/baseline/test --language java --tool all --verbose

run-generate-baseline-report:
	@echo "Generating baseline analysis report..."
	@$(VENV_ACTIVATE) $(PYTHON) scripts/baseline/generate_report.py
	@echo ""
	@echo "✅ Report generated in results/reports/"
	@echo "   - Visualizations: baseline_*.png (3 files)"
	@echo "   - Data exports: baseline_*.csv (2 files)"
	@echo "   - LaTeX table: baseline_summary.tex"
	@echo "   - Text report: baseline_report.txt"
	@echo "   - Summary: BASELINE_REPORT.md"

run-benchmark-verify:
	@echo "Verifying benchmarking infrastructure..."
	@$(VENV_ACTIVATE) $(PYTHON) -c "\
from src.utils.benchmark_utils import calculate_metrics, build_confusion_matrix, per_smell_breakdown, \
    statistical_tests, latency_profiler, ResourceMonitor, confidence_calibration, \
    rag_retrieval_quality, cache_hit_rate, validation_failure_rate, \
    calculate_hallucination_rate, compare_tools, save_metrics; \
print('✓ benchmark_utils — all functions imported'); \
from src.utils.result_exporter import to_latex_table, to_csv, export_predictions_csv, \
    plot_f1_comparison, plot_per_smell_heatmap, plot_confusion_matrix, \
    plot_latency_comparison, plot_confidence_calibration, plot_resource_usage, plot_delta_f1; \
print('✓ result_exporter — all functions imported'); \
print('✅ Benchmarking infrastructure verified')"

# ============================================================
# BASELINE TOOLS — Static Analysis (scripts/baseline/)
# ============================================================

# Full installation: Docker, SonarQube, PMD, Checkstyle, SpotBugs, ESLint, pylint, flake8
baseline-install:
	@bash scripts/baseline/install.sh

# Verify-only mode (no installs, just check what's available)
baseline-install-verify:
	@bash scripts/baseline/install.sh --verify

# Run Java baseline tools (PMD, Checkstyle, SpotBugs, SonarQube, IntelliJ)
baseline-java:
	@bash scripts/baseline/run.sh --language java --dir $(DIR) --timeout $(TIMEOUT) $(if $(TOOLS),--tools $(TOOLS)) $(if $(VERBOSE),--verbose)

# Run Python baseline tools (pylint, flake8)
baseline-python:
	@bash scripts/baseline/run.sh --language python --dir $(DIR) --timeout $(TIMEOUT) $(if $(TOOLS),--tools $(TOOLS)) $(if $(VERBOSE),--verbose)

# Run JavaScript baseline tools (ESLint)
baseline-js:
	@bash scripts/baseline/run.sh --language javascript --dir $(DIR) --timeout $(TIMEOUT) $(if $(TOOLS),--tools $(TOOLS)) $(if $(VERBOSE),--verbose)

# Auto-detect language and run all applicable tools
baseline-run:
	@bash scripts/baseline/run.sh --dir $(DIR) --timeout $(TIMEOUT) $(if $(VERBOSE),--verbose)

# Generate baseline analysis report (summary tables, heatmaps, LaTeX, CSV)
baseline-report:
	@echo "Generating baseline analysis report..."
	@$(VENV_ACTIVATE) $(PYTHON) scripts/baseline/generate_report.py
	@echo ""
	@echo "✅ Report generated in results/reports/"

# Verify benchmarking Python modules
baseline-verify:
	@$(VENV_ACTIVATE) $(PYTHON) -c "\
from src.utils.benchmark_utils import calculate_metrics, build_confusion_matrix, per_smell_breakdown, \
    statistical_tests, latency_profiler, ResourceMonitor; \
print('✓ benchmark_utils OK'); \
from src.utils.result_exporter import to_latex_table, to_csv, plot_f1_comparison; \
print('✓ result_exporter OK'); \
print('✅ Baseline modules verified')"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
	@echo "✓ Cache cleaned"

# ============================================================
# LLM Experiment Commands
# ============================================================

llm-baseline-run:
	@if [ -z "$(DIR)" ]; then DIR=data/datasets/marv/; fi
	@if [ -z "$(MODEL)" ]; then MODEL=llama3:8b; fi
	@echo "Running LLM Baseline Experiment"
	@echo "  Model: $(MODEL)"
	@echo "  Input: $(DIR)"
	@echo "  Temperature: 0.1"
	@echo "  Workers: 1 (M4 Pro optimized)"
	@echo ""
	@$(VENV_ACTIVATE) $(PYTHON) scripts/run_experiment.py \
		--experiment-type baseline \
		--input $(DIR) \
		--model $(MODEL) \
		--temperature 0.1 \
		--top-p 0.9 \
		--seed 42 \
		--workers 1 \
		--batch-size 1 \
		--verbose
	@echo "✅ Baseline experiment completed. Results in results/predictions/llm_vanilla/"

llm-rag-run:
	@if [ -z "$(DIR)" ]; then DIR=data/datasets/marv/; fi
	@if [ -z "$(MODEL)" ]; then MODEL=llama3:8b; fi
	@if [ -z "$(K)" ]; then K=5; fi
	@echo "Running LLM RAG Experiment"
	@echo "  Model: $(MODEL)"
	@echo "  Input: $(DIR)"
	@echo "  Top-K: $(K)"
	@echo "  Temperature: 0.1"
	@echo "  Workers: 1 (M4 Pro optimized)"
	@echo ""
	@$(VENV_ACTIVATE) $(PYTHON) scripts/run_experiment.py \
		--experiment-type rag \
		--input $(DIR) \
		--model $(MODEL) \
		--top-k $(K) \
		--temperature 0.1 \
		--top-p 0.9 \
		--seed 42 \
		--workers 1 \
		--batch-size 1 \
		--verbose
	@echo "✅ RAG experiment completed. Results in results/predictions/llm_rag/"

llm-ablation-run:
	@if [ -z "$(DIR)" ]; then DIR=data/datasets/marv/; fi
	@echo "Running Ablation Study"
	@echo "  Input: $(DIR)"
	@echo "  Studies: RAG hyperparameters, embedding models, prompt variants"
	@echo "  Method: 5-fold cross-validation"
	@echo "  Workers: 1 (M4 Pro optimized)"
	@echo ""
	@$(VENV_ACTIVATE) $(PYTHON) scripts/run_experiment.py \
		--experiment-type ablation \
		--input $(DIR) \
		--workers 1 \
		--batch-size 1 \
		--verbose
	@echo "✅ Ablation study completed. Results in results/predictions/llm_rag/"
