"""Visualizations that each teach one idea.

This is where we most raise the bar over the upstream repo (whose plot.py was
three seaborn line charts). matplotlib-only, headless-safe (Agg), artifacts land
in runs/. Style mirrors prototypical-toy: fixed per-class colors, snapshot ->
png + gif for anything that "evolves".
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless by default; stage scripts flip to interactive on --live
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA

from hierarchy import (
    COARSE_NAMES,
    COARSE_TO_FINE,
    FINE_BLOCK_ORDER,
    FINE_NAMES,
    NUM_COARSE,
)

RUNS = "runs"


def _ensure_runs() -> None:
    os.makedirs(RUNS, exist_ok=True)


def coarse_colors() -> np.ndarray:
    """20 stable, distinct colors — one per superclass — reused everywhere."""
    return plt.get_cmap("tab20")(np.linspace(0, 1, NUM_COARSE))


# ---------------------------------------------------------------------------
# Stage 0: the tree we're exploiting.
# ---------------------------------------------------------------------------
def draw_tree(path: str = f"{RUNS}/hierarchy_tree.png") -> str:
    _ensure_runs()
    colors = coarse_colors()
    fig, ax = plt.subplots(figsize=(11, 13))
    y = 0.0
    for c in range(NUM_COARSE):
        children = COARSE_TO_FINE[c]
        # The parent node sits at the vertical midpoint of its children, and
        # every edge fans out from that point.
        cy = y - (len(children) - 1) * 0.5
        for f in children:
            ax.plot([1, 2], [cy, y], color=colors[c], lw=1.2)
            ax.text(2.05, y, FINE_NAMES[f], va="center", fontsize=7)
            y -= 1.0
        ax.scatter([1], [cy], s=60, color=colors[c], zorder=3)
        ax.text(0.95, cy, COARSE_NAMES[c], va="center", ha="right", fontsize=8, weight="bold")
    ax.set_xlim(-0.2, 3.2)
    ax.axis("off")
    ax.set_title("CIFAR-100 taxonomy: 20 superclasses -> 100 fine classes", weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Stages 1-3: block-structured confusion matrix.
# ---------------------------------------------------------------------------
def block_confusion(y_true_fine: np.ndarray, y_pred_fine: np.ndarray,
                    title: str, path: str) -> str:
    """The 100x100 confusion matrix, read in the taxonomy's basis.

    A confusion matrix C has C[i, j] = #(true i, predicted j). Relabelling the
    classes by a permutation is a similarity transform C -> P C P.T: applying P
    on the left permutes rows, P.T on the right permutes columns, and doing BOTH
    keeps the diagonal on the diagonal. No information is added or lost -- we are
    only choosing a better basis. Ordering by FINE_BLOCK_ORDER puts siblings
    adjacent, so each superclass becomes a 5x5 block on the diagonal.

    That makes the interesting distinction visible at a glance: mass inside a
    diagonal block is a within-superclass mix-up (calling a maple an oak), while
    mass off the blocks is a mistake that crossed the taxonomy (calling a maple a
    motorcycle). Those are not equally bad errors, and a flat accuracy number
    cannot tell them apart. Suppressing the off-block mass is precisely the job
    of the hierarchy losses.

    We build C by accumulation rather than sklearn's confusion_matrix so the
    permutation is applied while counting -- `pos` maps a class id to its row in
    the new basis, which is P applied index-wise instead of materializing a
    100x100 permutation matrix and doing two matmuls for the same result.
    """
    _ensure_runs()
    pos = {fine: i for i, fine in enumerate(FINE_BLOCK_ORDER)}
    cm = np.zeros((100, 100))
    for t, p in zip(y_true_fine, y_pred_fine):
        cm[pos[int(t)], pos[int(p)]] += 1

    # Row-normalize: row i becomes the conditional distribution P(predicted | true=i),
    # so every row sums to 1 and rare and common classes are comparable. clip(...,1,None)
    # guards the 0/0 for any class absent from this split -- without it those rows
    # would be NaN and imshow would render them as holes.
    cm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="magma", vmin=0, vmax=0.5)
    for k in range(0, 101, 5):  # superclass block gridlines
        ax.axhline(k - 0.5, color="cyan", lw=0.4, alpha=0.5)
        ax.axvline(k - 0.5, color="cyan", lw=0.4, alpha=0.5)
    ax.set_title(title, weight="bold")
    ax.set_xlabel("predicted (fine, grouped by superclass)")
    ax.set_ylabel("true (fine, grouped by superclass)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalized rate")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Stages 2-4: metric curves over epochs.
# ---------------------------------------------------------------------------
def plot_curves(history: dict[str, list[float]], title: str, path: str) -> str:
    """history maps a series label -> per-epoch values; all plotted vs epoch."""
    _ensure_runs()
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, ys in history.items():
        ax.plot(range(1, len(ys) + 1), ys, marker="o", ms=3, label=label)
    ax.set_xlabel("epoch")
    ax.set_title(title, weight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_violation_comparison(series: dict[str, list[float]],
                              path: str = f"{RUNS}/violation_comparison.png") -> str:
    """The Stage-4 money shot: violation-rate curves for {lloss only, their dloss,
    soft loss, head agreement} on one axis."""
    return plot_curves(series, "Hierarchy-violation rate by loss (lower = respects the tree)", path)


# ---------------------------------------------------------------------------
# Stage 5: watch the representation organize by superclass.
# ---------------------------------------------------------------------------
class EmbeddingEvolution:
    """Collect (epoch, features, coarse_labels) snapshots, then render a GIF of
    the 2-D PCA colored by superclass + a static centroid-trajectory PNG.

    Same pattern as prototypical-toy's prototype_evolution/gif, applied to a
    real 20-way taxonomy. PCA (not UMAP) keeps deps to what's already installed.
    """

    def __init__(self, max_points: int = 2000):
        self._raw: list[tuple[int, np.ndarray, np.ndarray]] = []
        self.max_points = max_points

    def record(self, epoch: int, feats: np.ndarray, coarse: np.ndarray) -> None:
        """Stash this epoch's 256-D features; projection happens once, at the end."""
        if feats.shape[0] > self.max_points:
            # Seeded generator so the same points are tracked in every frame; a
            # fresh random subset per epoch would make points flicker in and out.
            idx = np.random.default_rng(0).choice(feats.shape[0], self.max_points, replace=False)
            feats, coarse = feats[idx], coarse[idx]
        self._raw.append((epoch, feats, coarse))

    @property
    def snaps(self) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """Every snapshot projected to 2-D in ONE shared PCA basis.

        PCA centres the data and takes the top two eigenvectors of its covariance
        matrix -- the two orthogonal directions of greatest variance -- then
        projects onto them. It is the best possible 2-D *linear* summary in the
        least-squares sense (Eckart-Young). We prefer it to t-SNE or UMAP because
        it is a fixed linear map: any clustering you see was already in the
        features, not manufactured by a nonlinear embedding that can carve
        clusters out of noise.

        Two choices matter. First, one basis for every frame: re-fitting per epoch
        would choose new axes each time, so the cloud would spin between frames and
        we could not tell real movement from a rotating coordinate system. Second,
        the basis is fitted on the FINAL snapshot, not the first. At random init
        the directions of greatest variance are arbitrary; the superclass structure
        the network later learns can lie almost entirely outside them, and the GIF
        would show the trained clusters collapsed on top of each other. Fitting on
        the trained features picks the axes where that structure lives, then plays
        the earlier epochs back in those same axes.
        """
        pca = PCA(n_components=2).fit(self._raw[-1][1])
        return [(ep, pca.transform(f), c) for ep, f, c in self._raw]

    def _centroids(self, emb: np.ndarray, coarse: np.ndarray) -> np.ndarray:
        cs = np.full((NUM_COARSE, 2), np.nan)
        for c in range(NUM_COARSE):
            m = coarse == c
            if m.any():
                cs[c] = emb[m].mean(0)
        return cs

    def save_gif(self, path: str = f"{RUNS}/embedding_evolution.gif", fps: int = 2) -> str:
        _ensure_runs()
        colors = coarse_colors()
        snaps = self.snaps
        all_xy = np.concatenate([e for _, e, _ in snaps])
        xlim = (all_xy[:, 0].min(), all_xy[:, 0].max())
        ylim = (all_xy[:, 1].min(), all_xy[:, 1].max())
        fig, ax = plt.subplots(figsize=(7, 7))

        def frame(i: int):
            ax.clear()
            ep, emb, coarse = snaps[i]
            ax.scatter(emb[:, 0], emb[:, 1], c=colors[coarse], s=6, alpha=0.6, linewidths=0)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_title(f"penultimate features (PCA), colored by superclass — epoch {ep}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        anim = FuncAnimation(fig, frame, frames=len(snaps), interval=1000 // fps)
        anim.save(path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        return path

    def save_trajectory(self, path: str = f"{RUNS}/embedding_trajectory.png") -> str:
        """Superclass centroids: o = first epoch, * = last, line = the path between."""
        _ensure_runs()
        colors = coarse_colors()
        fig, ax = plt.subplots(figsize=(7, 7))
        cents = [self._centroids(e, c) for _, e, c in self.snaps]
        for cidx in range(NUM_COARSE):
            track = np.array([cent[cidx] for cent in cents])
            ax.plot(track[:, 0], track[:, 1], color=colors[cidx], lw=1, alpha=0.7)
            ax.scatter(track[0, 0], track[0, 1], color=colors[cidx], marker="o", s=30)
            ax.scatter(track[-1, 0], track[-1, 1], color=colors[cidx], marker="*", s=140,
                       edgecolors="k", linewidths=0.4)
        ax.set_title("superclass centroid trajectories (o start, * final)", weight="bold", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
