"""Keep the complete Python examples executable with the current public API."""

import re
from pathlib import Path

import pytest
from topic_boundaries import Settings


@pytest.mark.integration
@pytest.mark.parametrize(
    "name",
    ["text_corpus", "precomputed_euclidean", "compare_methods", "visualize"],
)
def test_documented_example_runs(name, redis_url, isolated_schema, monkeypatch):
    redis = pytest.importorskip("redis")
    try:
        redis.from_url(redis_url).ping()
    except redis.RedisError:
        pytest.skip("Redis Stack not reachable")

    settings = Settings.from_config(schema_path=isolated_schema())
    monkeypatch.setattr(Settings, "from_config", lambda: settings)
    page = Path(__file__).resolve().parents[1] / "docs" / "examples" / f"{name}.md"
    block = re.search(r"```python\n(.*?)\n```", page.read_text(encoding="utf-8"), re.S)
    assert block is not None, f"missing runnable Python block in {page}"
    exec(compile(block.group(1), str(page), "exec"), {"__name__": "__main__"})
