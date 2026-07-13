"""Embedders turn Datapoints into a float32 (n, dim) matrix.

The Embedder protocol is the seam that lets the pipeline work on *any* vector
space: use HFTextEmbedder to embed text bodies, or PrecomputedEmbedder to pass
through vectors you already have (image/audio/arbitrary features).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from topic_boundaries.documents import Datapoint


@runtime_checkable
class Embedder(Protocol):
    def embed(self, datapoints: list[Datapoint]) -> np.ndarray:
        """Return a float32 array of shape (len(datapoints), dims)."""
        ...

    @property
    def dims(self) -> int | None:
        """Embedding dimensionality, or None if unknown until embed() runs."""
        ...


class HFTextEmbedder:
    """Embed ``datapoint.body`` with a sentence-transformers model (the default)."""

    def __init__(self, model: str, *, batch_size: int = 64):
        # Imported lazily so precomputed-vector users don't pay the heavy import.
        from redisvl.utils.vectorize import HFTextVectorizer

        self._vectorizer = HFTextVectorizer(model=model)
        self._batch_size = batch_size

    def embed(self, datapoints: list[Datapoint]) -> np.ndarray:
        texts = [d.body for d in datapoints]
        return np.asarray(
            self._vectorizer.embed_many(
                contents=texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > self._batch_size,
            ),
            dtype=np.float32,
        )

    @property
    def dims(self) -> int | None:
        return self._vectorizer.dims


class PrecomputedEmbedder:
    """Pass through ``datapoint.vector`` — no embedding, any vector space.

    Every datapoint must carry a ``vector`` of the same length.
    """

    def __init__(self) -> None:
        self._dims: int | None = None

    def embed(self, datapoints: list[Datapoint]) -> np.ndarray:
        if not datapoints:
            raise ValueError("PrecomputedEmbedder got no datapoints.")
        missing = [d.doc_id for d in datapoints if d.vector is None]
        if missing:
            raise ValueError(
                f"{len(missing)} datapoint(s) have no precomputed vector "
                f"(first: {missing[0]!r}). Set Datapoint.vector for every point."
            )
        vectors = np.asarray([np.asarray(d.vector, dtype=np.float32) for d in datapoints])
        if vectors.ndim != 2:
            dims = {np.asarray(d.vector).shape for d in datapoints}
            raise ValueError(f"Precomputed vectors have inconsistent shapes: {dims}.")
        self._dims = int(vectors.shape[1])
        return vectors.astype(np.float32)

    @property
    def dims(self) -> int | None:
        return self._dims
