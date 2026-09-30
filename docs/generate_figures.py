"""Regenerate the original 2D method diagrams used by the Sphinx docs.

Run from the repository root with `pip install -e ".[viz]"`.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from topic_boundaries import Datapoint, voronoi_rankings
from topic_boundaries.visualization import export_png


OUTPUT = Path("docs/_static/figures")
OUTPUT.mkdir(parents=True, exist_ok=True)

A = np.array([[-3.2, -0.9], [-3.0, 0.5], [-2.3, -0.5], [-2.0, 0.7],
              [-1.8, -1.1], [-1.1, 0.2], [-0.2, 0.0]], dtype=float)
B = np.array([[0.9, 0.1], [1.4, -0.8], [1.6, 0.9], [2.2, -0.2],
              [2.6, 0.7], [2.8, -0.9], [3.1, 0.1]], dtype=float)
CA = np.array([-2.0, 0.0])
CB = np.array([2.0, 0.0])
BLUE = "#277DA1"
ORANGE = "#F8961E"
RED = "#C62828"


def base(title: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(A[:, 0], A[:, 1], color=BLUE, s=58, label="cluster A", zorder=3)
    ax.scatter(B[:, 0], B[:, 1], color=ORANGE, s=58, label="cluster B", zorder=3)
    ax.scatter([CA[0], CB[0]], [CA[1], CB[1]], marker="X", s=160,
               color="black", label="centroids", zorder=4)
    ax.set(xlim=(-4.0, 3.8), ylim=(-1.7, 1.6),
           xlabel="embedding dimension 1", ylabel="embedding dimension 2",
           title=title)
    ax.grid(alpha=0.15)
    ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    return fig, ax


def save(fig, name: str):
    fig.savefig(OUTPUT / name, dpi=160)
    plt.close(fig)


fig, ax = base("Farthest from centroid")
far = A[np.argmax(np.linalg.norm(A - CA, axis=1))]
ax.plot([CA[0], far[0]], [CA[1], far[1]], color=RED, linewidth=2.5)
ax.scatter(*far, color=RED, s=115, edgecolor="black", zorder=5)
ax.annotate("largest within-cluster distance", far, xytext=(-0.8, -1.45),
            arrowprops={"arrowstyle": "->", "color": RED}, color=RED, fontsize=9)
save(fig, "farthest.png")

fig, ax = base("Convex hull of one cluster")
hull = ConvexHull(A)
ring = A[np.r_[hull.vertices, hull.vertices[0]]]
ax.plot(ring[:, 0], ring[:, 1], color=RED, linewidth=2.4)
ax.scatter(A[hull.vertices, 0], A[hull.vertices, 1], color=RED,
           s=100, edgecolor="black", zorder=5)
ax.annotate("hull vertices", (-3.2, -0.9), xytext=(-1.0, -1.45),
            arrowprops={"arrowstyle": "->", "color": RED}, color=RED, fontsize=9)
save(fig, "convex-hull.png")

fig, ax = base("Near a K-means decision boundary")
edge = A[-1]
ax.axvline(0, color=RED, linestyle="--", linewidth=2, label="decision boundary")
ax.plot([CA[0], edge[0]], [CA[1], edge[1]], color=BLUE, linewidth=2)
ax.plot([edge[0], CB[0]], [edge[1], CB[1]], color=ORANGE, linewidth=2)
ax.scatter(*edge, color=RED, s=115, edgecolor="black", zorder=5)
ax.annotate("similar distance to both centroids", edge, xytext=(-3.7, -1.45),
            arrowprops={"arrowstyle": "->", "color": RED}, color=RED, fontsize=9)
save(fig, "voronoi.png")

fig, ax = base("Nearest document outside a cluster")
neighbor = B[np.argmin(np.linalg.norm(B - CA, axis=1))]
ax.annotate("", xy=neighbor, xytext=CA,
            arrowprops={"arrowstyle": "->", "color": RED, "lw": 2.5})
ax.scatter(*neighbor, color=RED, s=115, edgecolor="black", zorder=5)
ax.annotate("nearest point in cluster B", neighbor, xytext=(0.2, -1.4),
            arrowprops={"arrowstyle": "->", "color": RED}, color=RED, fontsize=9)
save(fig, "cross-boundary.png")

points = np.vstack([A, B]).astype(np.float32)
labels = np.array([0] * len(A) + [1] * len(B))
centroids = np.stack([CA, CB]).astype(np.float32)
datapoints = [Datapoint(str(i), vector=point) for i, point in enumerate(points)]
hits = voronoi_rankings(datapoints, points, labels, centroids, 2, top_n_per_cluster=1)
export_png(labels, {int(hit.doc_id) for hit in hits}, points,
           centroids, OUTPUT / "library-plot.png",
           title="Library output: highlighted boundary candidates")
