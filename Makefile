.PHONY: help setup install notebook test clean poetry-fix verify fetch-data reports lint check run-notebooks

# Default target
help:
	@echo "CLOB Risk Modelling - Available commands:"
	@echo "  make setup          - Configure Poetry and install dependencies"
	@echo "  make notebook       - Launch Jupyter notebook server"
	@echo "  make run-notebooks  - Execute all notebooks with current data"
	@echo "  make fetch-data     - Fetch latest market data from APIs"
	@echo "  make reports        - Generate risk reports from notebooks"
	@echo "  make test           - Run test suite"
	@echo "  make lint           - Run code linting"
	@echo "  make check          - Run type checking and linting"
	@echo "  make clean          - Clean generated files and cache"
	@echo "  make poetry-fix     - Fix Poetry/pyenv issues"
	@echo "  make verify         - Verify installation and dependencies"

# Setup Poetry and install dependencies
setup:
	@echo "🔧 Configuring Poetry..."
	poetry config virtualenvs.in-project true
	@echo "📦 Installing dependencies..."
	poetry install --no-root
	@echo "📊 Installing Jupyter kernel..."
	poetry run python -m ipykernel install --user --name clob-risk --display-name "CLOB Risk"
	@echo "✅ Setup complete!"

# Launch Jupyter notebook
notebook:
	@echo "📊 Creating directories if needed..."
	@mkdir -p risk-model/notebooks
	@mkdir -p risk-model/src/risk_model
	@mkdir -p risk-model/config
	@mkdir -p data/raw
	@mkdir -p data/processed
	@mkdir -p reports
	@echo "🚀 Launching Jupyter notebook..."
	cd risk-model && poetry run jupyter notebook --port=8888

# Fetch latest market data
fetch-data:
	@echo "📈 Fetching latest market data..."
	@mkdir -p data/raw/binance
	@mkdir -p data/raw/uniswap
	@echo "Running data fetch scripts..."
	@if [ -f risk-model/src/risk_model/binance_data.py ]; then \
		cd risk-model && poetry run python -c "import sys; sys.path.append('src'); from risk_model.binance_data import fetch_all_data; fetch_all_data()"; \
	else \
		echo "⚠️  Data modules not yet implemented"; \
	fi

# Generate risk reports
reports:
	@echo "📊 Generating risk reports..."
	@mkdir -p reports/$(shell date +%Y%m%d)
	@echo "Converting notebooks to HTML reports..."
	@for notebook in risk-model/notebooks/*.ipynb; do \
		if [ -f "$$notebook" ]; then \
			echo "Processing $$notebook..."; \
			poetry run jupyter nbconvert --to html --output-dir reports/$(shell date +%Y%m%d) "$$notebook"; \
		fi; \
	done
	@echo "✅ Reports generated in reports/$(shell date +%Y%m%d)/"

# Run tests
test:
	@echo "🧪 Running tests..."
	@if [ -d tests ]; then \
		poetry run pytest tests/ -v; \
	else \
		echo "⚠️  No tests directory found yet"; \
	fi

# Lint code
lint:
	@echo "🔍 Running code linting..."
	poetry run ruff check risk-model/src/ --fix

# Type checking and linting
check:
	@echo "🔍 Running type checking and linting..."
	poetry run ruff check risk-model/src/
	poetry run mypy risk-model/src/ --ignore-missing-imports

# Clean up generated files
clean:
	@echo "🧹 Cleaning up..."
	rm -rf .venv/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .ipynb_checkpoints/
	rm -rf risk-model/notebooks/.ipynb_checkpoints/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "✅ Cleanup complete!"

# Fix Poetry issues
poetry-fix:
	@echo "🔧 Fixing Poetry configuration..."
	poetry cache clear pypi --all || true
	poetry env remove --all || true
	poetry config virtualenvs.prefer-active-python false
	poetry config experimental.system-git-client true
	@echo "✅ Poetry cache cleared. Run 'make setup' to reinstall."

# Install Poetry if not installed
install-poetry:
	@if ! command -v poetry &> /dev/null; then \
		echo "📦 Installing Poetry..."; \
		curl -sSL https://install.python-poetry.org | python3 -; \
		echo "✅ Poetry installed! Add $$HOME/.local/bin to your PATH"; \
	else \
		echo "✅ Poetry is already installed"; \
	fi

# Verify installation
verify:
	@echo "🔍 Verifying installation..."
	@echo ""
	@echo "Python version:"
	@poetry run python --version
	@echo ""
	@echo "Installed packages:"
	@poetry show
	@echo ""
	@echo "Checking required directories:"
	@if [ -d risk-model ]; then echo "✅ risk-model/ directory exists"; else echo "⚠️  risk-model/ directory missing"; fi
	@if [ -d risk-model/notebooks ]; then echo "✅ notebooks/ directory exists"; else echo "⚠️  notebooks/ directory missing"; fi
	@if [ -d risk-model/src/risk_model ]; then echo "✅ src/risk_model/ directory exists"; else echo "⚠️  src/risk_model/ directory missing"; fi
	@if [ -d risk-model/config ]; then echo "✅ config/ directory exists"; else echo "⚠️  config/ directory missing"; fi
	@echo ""
	@echo "Checking Jupyter kernel:"
	@poetry run jupyter kernelspec list | grep clob-risk || echo "⚠️  CLOB Risk kernel not installed"
	@echo ""
	@echo "✅ Verification complete!"

# Run all notebooks in sequence
run-notebooks:
	@echo "🚀 Running all risk model notebooks..."
	@mkdir -p risk-model/notebooks/executed
	@echo "📊 Executing data exploration notebook..."
	@cd risk-model && poetry run jupyter nbconvert --to notebook --execute notebooks/00_data_exploration.ipynb --output executed/00_data_exploration_$(shell date +%Y%m%d).ipynb
	@echo "📊 Executing market risk limits notebook..."
	@cd risk-model && poetry run jupyter nbconvert --to notebook --execute notebooks/01_market_risk_limits.ipynb --output executed/01_market_risk_limits_$(shell date +%Y%m%d).ipynb
	@echo "📊 Executing stress testing notebook..."
	@cd risk-model && poetry run jupyter nbconvert --to notebook --execute notebooks/02_stress_testing.ipynb --output executed/02_stress_testing_$(shell date +%Y%m%d).ipynb
	@echo "📊 Executing manipulation simulation notebook..."
	@cd risk-model && poetry run jupyter nbconvert --to notebook --execute notebooks/03_manipulation_sims.ipynb --output executed/03_manipulation_sims_$(shell date +%Y%m%d).ipynb
	@echo "📊 Executing risk summary notebook..."
	@cd risk-model && poetry run jupyter nbconvert --to notebook --execute notebooks/risk_summary.ipynb --output executed/risk_summary_$(shell date +%Y%m%d).ipynb
	@echo "✅ All notebooks executed successfully! Results in risk-model/notebooks/executed/"