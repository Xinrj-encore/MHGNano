# Contributing to HyperGraph Meta-PPI

Thank you for your interest in contributing!

## Development Setup

1. Fork the repository
2. Clone your fork
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

## Code Style

- Follow PEP 8
- Use black for formatting: `black src/`
- Use flake8 for linting: `flake8 src/`

## Testing

Run tests with pytest:
```bash
pytest tests/
```

## Pull Request Process

1. Create a feature branch
2. Make your changes
3. Add tests if applicable
4. Update documentation
5. Submit a pull request

## Reporting Issues

Please use GitHub Issues to report bugs or request features.
