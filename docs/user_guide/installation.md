# Installation

## Requirements

- Python 3.10 or newer
- Redis Stack with RediSearch and vector indexing

From the repository root, install the package and start the included Redis service:

```bash
pip install -e ".[dev]"
docker compose up -d
```

The default connection is `redis://localhost:6379/0`. You can change `redis_url` in `config.yml` or set `REDIS_URL` in your shell. The environment variable takes precedence.

## Optional features

```bash
pip install -e ".[corpus]"  # PDF loading with pypdf
pip install -e ".[viz]"     # Plotly, Matplotlib, and UMAP visualization
```

The base install includes the default sentence-transformers embedder. Its model weights may download on first use. Use `PrecomputedEmbedder` if you already have vectors.

## Preview this documentation locally

```bash
pip install -e ".[docs]"
sphinx-build -b html -W --keep-going docs docs/_build/html
python -m http.server --bind 127.0.0.1 8000 --directory docs/_build/html
```

Open `http://127.0.0.1:8000/`. The generated site stays under `docs/_build/`, which Git ignores.
