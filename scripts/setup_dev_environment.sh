#!/bin/bash
# Development environment setup script for Laser Ablation Lab

set -e

echo "🚀 Setting up Laser Ablation Lab development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and install build tools
echo "⬆️ Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel

# Install development dependencies
echo "📚 Installing development dependencies..."
if [ -f "requirements-dev.txt" ]; then
    pip install -r requirements-dev.txt
else
    pip install -e ".[dev]"
fi

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install

# Run initial code quality checks
echo "🔍 Running initial code quality checks..."
black --check src/ tests/ || echo "⚠️ Run 'black src/ tests/' to fix formatting"
ruff check src/ tests/ || echo "⚠️ Fix linting issues before committing"

# Create required directories
echo "📁 Creating required directories..."
mkdir -p data/{raw,processed,interim,external}
mkdir -p results/{simulations,plots,reports}
mkdir -p logs

# Generate sample configuration
echo "⚙️ Generating sample configuration..."
cat > config/local/development.yaml << 'CONFIG_EOF'
# Development configuration for Laser Ablation Lab
simulation:
  default_material: "basalt"
  default_grid_size: 100
  max_time_steps: 10000
  convergence_tolerance: 1e-8

output:
  data_directory: "data"
  results_directory: "results"
  log_level: "INFO"
  save_intermediate_results: true

physics:
  use_adaptive_time_stepping: true
  enable_parallelization: false
  memory_limit_mb: 8000

visualization:
  backend: "matplotlib"
  interactive_plots: true
  save_animations: false
CONFIG_EOF

# Test basic imports
echo "🧪 Testing basic package imports..."
python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    import ablab
    import ablab.constants
    import ablab.units
    print('✅ All core modules imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

# Setup Jupyter kernel
echo "📓 Setting up Jupyter kernel..."
python3 -m ipykernel install --user --name=ablation-lab --display-name="Laser Ablation Lab"

# Generate example notebook
echo "📔 Generating example notebook..."
cat > examples/00_quick_start.ipynb << 'NOTEBOOK_EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Laser Ablation Lab - Quick Start\n",
    "\n",
    "This notebook demonstrates basic usage of the Laser Ablation Lab framework.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Import laser ablation lab modules\n",
    "from ablab.constants import get_material, PhysicalConstants\n",
    "from ablab.units import Quantity, meters, watts, seconds\n",
    "\n",
    "print(\"🚀 Laser Ablation Lab Quick Start\")\n",
    "print(f\"Speed of light: {PhysicalConstants.SPEED_OF_LIGHT} m/s\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Load material properties\n",
    "basalt = get_material('basalt')\n",
    "print(f\"Material: {basalt['name']}\")\n",
    "print(f\"Density: {basalt['density']} kg/m³\")\n",
    "print(f\"Melting point: {basalt['melting_point']} K\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Demonstrate unit system\n",
    "length = meters(1.5, uncertainty=0.1)\n",
    "power = watts(1e6)\n",
    "time = seconds(10)\n",
    "\n",
    "energy = power * time\n",
    "print(f\"Length: {length}\")\n",
    "print(f\"Power: {power}\")\n",
    "print(f\"Energy: {energy}\")\n",
    "\n",
    "# Convert units\n",
    "length_cm = length.to('cm')\n",
    "print(f\"Length in cm: {length_cm}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Laser Ablation Lab",
   "language": "python",
   "name": "ablation-lab"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
NOTEBOOK_EOF

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Start Jupyter Lab: jupyter lab"
echo "  3. Open examples/00_quick_start.ipynb"
echo "  4. Run tests: pytest tests/"
echo "  5. Check code quality: black src/ tests/ && ruff check src/ tests/"
echo ""
echo "For Docker development:"
echo "  docker-compose up development"
echo "  # Access Jupyter Lab at http://localhost:8888"
echo ""
