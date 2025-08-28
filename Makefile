# Makefile for vision2time project
.PHONY: help setup install install-pip status benchmark clean clean-env

# Colors for terminal output
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
BLUE=\033[0;34m
NC=\033[0m # No Color

# Default environment name
ENV_NAME=vision2time

# Help target
help:
	@echo "$(BLUE)vision2time Makefile Commands$(NC)"
	@echo "=========================="
	@echo ""
	@echo "$(GREEN)Setup & Installation:$(NC)"
	@echo "  make setup          - Create conda environment and install dependencies"
	@echo "  make install        - Install package in development mode"
	@echo "  make install-pip    - Alternative pip-only installation"
	@echo ""
	@echo "$(GREEN)Data & Testing:$(NC)"
	@echo "  make status         - Check environment and data status"
	@echo ""
	@echo "$(GREEN)Benchmarking:$(NC)"
	@echo "  make benchmark      - Run full benchmark"
	@echo ""
	@echo "$(GREEN)Utilities:$(NC)"
	@echo "  make clean          - Clean build artifacts and results"
	@echo "  make clean-env      - Remove conda environment"

# Setup conda environment
setup:
	@echo "$(GREEN)Setting up vision2time environment...$(NC)"
	@if conda env list | grep -q "^$(ENV_NAME) "; then \
		echo "$(YELLOW)Environment $(ENV_NAME) already exists$(NC)"; \
		echo "$(BLUE)Updating environment...$(NC)"; \
		conda env update -f environment.yaml; \
	else \
		echo "$(BLUE)Creating new environment...$(NC)"; \
		conda env create -f environment.yaml; \
	fi
	@echo "$(GREEN)Environment setup complete!$(NC)"
	@echo ""
	@echo "$(YELLOW)To activate the environment manually:$(NC)"
	@echo "  conda activate $(ENV_NAME)"

# Install in development mode
install:
	@echo "$(GREEN)Installing package in development mode...$(NC)"
	@if command -v conda >/dev/null 2>&1; then \
		conda run -n $(ENV_NAME) pip install -e .; \
	else \
		pip install -e .; \
	fi
	@echo "$(GREEN)Development installation complete!$(NC)"

# Alternative pip installation
install-pip:
	@echo "$(GREEN)Installing with pip...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)Pip installation complete!$(NC)"

# Check status
status:
	@echo "$(GREEN)vision2time Status Check$(NC)"
	@echo "======================="
	@echo ""
	@echo "$(BLUE)Environment:$(NC)"
	@if command -v conda >/dev/null 2>&1; then \
		if conda env list | grep -q "^$(ENV_NAME) "; then \
			echo "Environment '$(ENV_NAME)' exists"; \
			conda run -n $(ENV_NAME) python --version; \
		else \
			echo "$(RED)Conda environment '$(ENV_NAME)' not found$(NC)"; \
			echo "Run: make setup"; \
		fi \
	else \
		echo "$(YELLOW)Conda not found, checking system Python$(NC)"; \
		python --version; \
	fi
	@echo ""
	@echo "$(BLUE)Dependencies:$(NC)"
	@if command -v conda >/dev/null 2>&1; then \
		conda run -n $(ENV_NAME) python -c "import torch; print('PyTorch:', torch.__version__)" 2>/dev/null || echo "$(RED)PyTorch not installed$(NC)"; \
		conda run -n $(ENV_NAME) python -c "import transformers; print('Transformers:', transformers.__version__)" 2>/dev/null || echo "$(RED)Transformers not installed$(NC)"; \
	else \
		python -c "import torch; print('PyTorch:', torch.__version__)" 2>/dev/null || echo "$(RED)PyTorch not installed$(NC)"; \
		python -c "import transformers; print('Transformers:', transformers.__version__)" 2>/dev/null || echo "$(RED)Transformers not installed$(NC)"; \
	fi
	@echo ""
	@echo "$(BLUE)Data:$(NC)"
	@if [ -d "data/raw/Multivariate_ts" ]; then \
		dataset_count=$$(ls data/raw/Multivariate_ts/*.npy 2>/dev/null | wc -l); \
		if [ $$dataset_count -gt 0 ]; then \
			echo "Found $$dataset_count dataset(s)"; \
			ls data/raw/Multivariate_ts/; \
		else \
			echo "$(YELLOW)Data directory exists but no datasets found$(NC)"; \
		fi \
	else \
		echo "$(YELLOW)Data directory not found$(NC)"; \
	fi
	@echo ""
	@echo "$(BLUE)Results:$(NC)"
	@if [ -f "data/results/benchmark_results.csv" ]; then \
		echo "Results file exists"; \
		echo "Experiments: $$(tail -n +2 data/results/benchmark_results.csv | wc -l)"; \
	else \
		echo "$(YELLOW)No results found$(NC)"; \
	fi

# Run benchmark
benchmark:
	@echo "$(GREEN)Running benchmark...$(NC)"
	@if [ ! -d "data/raw/Multivariate_ts" ] || [ -z "$$(ls -A data/raw/Multivariate_ts 2>/dev/null)" ]; then \
		echo "$(YELLOW)No data found, creating test dataset...$(NC)"; \
		mkdir -p data/raw/Multivariate_ts; \
		if command -v conda >/dev/null 2>&1; then \
			conda run -n $(ENV_NAME) python main.py --mode create-test; \
		else \
			python main.py --mode create-test; \
		fi \
	fi
	@if command -v conda >/dev/null 2>&1; then \
		conda run -n $(ENV_NAME) python main.py --mode benchmark; \
	else \
		python main.py --mode benchmark; \
	fi
	@echo "$(GREEN)Benchmark complete!$(NC)"
	@if [ -f "data/results/benchmark_results.csv" ]; then \
		echo "$(BLUE)Results saved to: data/results/benchmark_results.csv$(NC)"; \
		echo "$(BLUE)Total experiments: $$(tail -n +2 data/results/benchmark_results.csv | wc -l)$(NC)"; \
	fi

# Clean build artifacts
clean:
	@echo "$(GREEN)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	find . -type d -name __pycache__ -delete
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	rm -rf data/results/*
	@echo "$(GREEN)Clean complete!$(NC)"

# Remove conda environment
clean-env:
	@echo "$(GREEN)Removing conda environment...$(NC)"
	@if conda env list | grep -q "^$(ENV_NAME) "; then \
		conda env remove -n $(ENV_NAME); \
		echo "$(GREEN)Environment removed!$(NC)"; \
	else \
		echo "$(YELLOW)Environment $(ENV_NAME) not found$(NC)"; \
	fi