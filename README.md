# lylexpy

WDBX is a modular Flask-based Python application and library providing a vector database, blockchain anchoring, and plugin platform. It includes:

- High-level `wdbx` Python package:
  - Synchronous `WDBX` client
  - Asynchronous `AsyncWDBX` client
  - Features: vector storage/search, metadata, transactions, blockchain operations, plugins, self-updates, and more
- Flask-based HTTP server (`app.py`) exposing JWT-protected API routes under `/api`
- Plugin ecosystem under `plugins/` with utilities for PDF conversion, web scraping, crypto, ML training, and integrations (Discord, Ollama, etc.)
- Utility package `plugin_utils/` for common validation, error handling, metrics, and health checks
- Comprehensive documentation generated with MkDocs (see below)

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Frostshake/lylexpy.git
   cd lylexpy
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and configure environment variables.
4. Run the Flask application:
   ```bash
   flask run
   ```

## Documentation

Detailed documentation is available via MkDocs. To view the documentation site locally:

```bash
pip install mkdocs mkdocstrings
mkdocs serve
``` 

Browse the documentation at `http://localhost:8000`.

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing.

## License

This project is licensed under the [MIT License](LICENSE).

## Development & Code Quality

Install development dependencies (including Black, isort, pytest, etc.):
```bash
pip install -e .[dev]
```

Run pre-commit checks (formats, linting, types):
```bash
pre-commit run --all-files
```

Automatically format code with Black:
```bash
black .
isort .
```


## Testing

Run the test suite with pytest:
```bash
pytest
``` 