# Contributing to Laser Ablation Lab

Thank you for your interest in contributing to the Laser Ablation Lab project! This project aims to provide open, modular simulation tools for studying laser ablation and laser-induced disruption of asteroids and comets.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/laser-ablation-lab.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements-dev.txt`
6. Install the package in development mode: `pip install -e .`

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest tests/`
4. Run linting: `ruff check src/ tests/`
5. Run formatting: `black src/ tests/`
6. Run type checking: `mypy src/`
7. Commit your changes with a descriptive message
8. Push to your fork and create a pull request

## Code Standards

### Python Style
- Follow PEP 8 guidelines
- Use Black for code formatting
- Use Ruff for linting
- Use type hints where appropriate
- Maximum line length: 88 characters (Black default)

### Naming Conventions
- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`
- Modules: `lowercase` or `snake_case`

### Documentation
- Use Google-style docstrings
- Include physics/engineering context in docstrings
- Provide units for all physical quantities
- Include references to scientific literature where applicable

### Testing
- Write unit tests for all new functions
- Include physics validation tests with known solutions
- Test boundary conditions and error cases
- Aim for >90% code coverage

## Physics and Engineering Guidelines

### Units and Conversions
- Use SI units internally unless otherwise specified
- Clearly document unit assumptions
- Provide unit conversion utilities
- Validate dimensional analysis

### Model Validation
- Compare against published experimental data
- Include uncertainty quantification
- Document model limitations and assumptions
- Provide references to source literature

### Performance Considerations
- Profile computational hotspots
- Use NumPy vectorization where possible
- Consider Numba for performance-critical loops
- Document computational complexity

## Submitting Changes

### Pull Request Process
1. Ensure all tests pass
2. Update documentation as needed
3. Add entries to CHANGELOG.md
4. Fill out the pull request template completely
5. Request review from maintainers

### Commit Messages
Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for adding tests
- `refactor:` for code refactoring
- `perf:` for performance improvements

Example: `feat: add momentum coupling coefficient calculation for vapor regime`

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Acknowledge contributions from others
- Maintain professional communication

## Questions and Support

- Open an issue for bugs or feature requests
- Use discussions for general questions
- Tag maintainers for urgent issues
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.
