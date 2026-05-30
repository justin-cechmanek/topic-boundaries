"""2D inspection of topic clusters, centroids, and boundary results."""

from src.visualization.core import boundary_highlight_indices, project_to_2d
from src.visualization.figure import build_plotly_figure

__all__ = [
    "boundary_highlight_indices",
    "build_plotly_figure",
    "project_to_2d",
]
