Python reference
================

Data and configuration
----------------------

.. autoclass:: topic_boundaries.Datapoint

.. autoclass:: topic_boundaries.Settings
   :members: from_config

.. autofunction:: topic_boundaries.datapoint_from_record

.. autofunction:: topic_boundaries.load_jsonl

Sources
-------

Each ``load()`` method returns ``list[Datapoint]``. PDF support requires the
``corpus`` extra; arXiv loaders make network requests.

.. autoclass:: topic_boundaries.JsonlSource
   :members: load

.. autoclass:: topic_boundaries.JsonSource
   :members: load

.. autoclass:: topic_boundaries.CsvSource
   :members: load

.. autoclass:: topic_boundaries.RecordsSource
   :members: load

.. autoclass:: topic_boundaries.TextFileSource
   :members: load

.. autoclass:: topic_boundaries.TextDirSource
   :members: load

.. autoclass:: topic_boundaries.PdfSource
   :members: load

.. autoclass:: topic_boundaries.ArxivSource
   :members: load

.. autoclass:: topic_boundaries.ArxivPaperSource
   :members: load

Embedding, clustering, and indexing
-----------------------------------

.. autoclass:: topic_boundaries.Embedder
   :members: embed, dims

.. autoclass:: topic_boundaries.HFTextEmbedder
   :members: embed, dims

.. autoclass:: topic_boundaries.PrecomputedEmbedder
   :members: embed, dims

.. autofunction:: topic_boundaries.embed_and_cluster_datapoints

.. autofunction:: topic_boundaries.index_clustered_datapoints

Boundary methods and results
----------------------------

.. autofunction:: topic_boundaries.max_distance_rankings

.. autoclass:: topic_boundaries.BoundaryHit

.. autofunction:: topic_boundaries.cross_boundary_hits_for_all_clusters

.. autoclass:: topic_boundaries.CrossBoundaryHit

.. autofunction:: topic_boundaries.boundary_doc_indices_per_cluster

.. autofunction:: topic_boundaries.voronoi_rankings

.. autoclass:: topic_boundaries.VoronoiBoundaryHit

.. autofunction:: topic_boundaries.nearest_to_centroid
