#!/usr/bin/env python3
"""scripts/plot_latent_tsne.py

Render the tactile latent analysis figures from the per-task `.npz`
artifacts produced by `scripts/extract_latents_for_tsne.py`.

This script is deliberately MODEL-FREE: it only consumes saved numpy
arrays, so it runs anywhere (CPU, no torch, no VAE) and can be smoke-
tested with `--synthetic`.

Scope (reframed 2026-07-16). The analysis proves exactly TWO claims about
the fused per-hand tactile latent the DiT actually consumes, and draws the
LEFT and RIGHT hands as SEPARATE figures (never pooled with an L/R marker):

  Claim 1 - task specificity.
    Fig 1L / Fig 1R: per-hand fused-latent t-SNE, 4 tasks colored.
    t-SNE is VISUALIZATION ONLY. The headline evidence is computed in the
    raw 128-D latent space, per hand:
      * silhouette_128d
      * stratified 5-fold kNN accuracy (fixed k=5), mean +/- std
      * multinomial logistic-probe accuracy, mean +/- std
    (the 2-D t-SNE silhouette is reported as a plotting diagnostic only).

  Claim 2 - per-finger sensitivity retained.
    Fig 2L / Fig 2R: per-hand leave-one-finger-out fusion shift, 5 bars,
    mean + 95% bootstrap CI, split by `tactile_hand_side`. Wording:
    the fused representation "remains sensitive to every finger / does not
    collapse any single finger" -- NOT "losslessly preserves".

Supplementary (off the main claim path):
  Fig 3L / Fig 3R: pre-fusion per-finger t-SNE, all tasks pooled, color =
    finger id. Shows the 5 finger streams entering the adapter are
    distinguishable. `--facet-by-task` optionally adds a per-task
    small-multiple; default OFF (no hand x task double-facet).

Diagnostic (NOT a paper figure, default not rendered):
  vision-vs-tactile modality separation. The data show
    d(mu_vision, mu_tactile) / intra_modal_spread >> 1, i.e. the two
    modalities occupy DISTINCT regions. This is recorded in summary.json
    under `diagnostics.vision_tactile_separation` as a modality-SEPARATION
    metric (higher = more separated, NOT overlap). No shared-manifold /
    alignment / consistency claim is made anywhere. Pass
    `--emit-diagnostic-separation` to also render the overlaid t-SNE PNG
    for internal inspection.

Per-task `.npz` schema (written by extract_latents_for_tsne.py):
  task              : () str
  vision            : (N_vis, D)   pooled RGB head-view latent
  tactile_hand      : (N_hand, D)  pooled fused per-hand latent
  tactile_hand_side : (N_hand,) int  hand index (0=left, 1=right, ...)
  tactile_finger    : (N_fing, D)  pooled per-finger latent
  finger_id         : (N_fing,) int  0..F-1
  finger_side       : (N_fing,) int  hand index for each finger row
  fusion_shift      : (N_hand, F) float  normalized per-finger shift
                      (row order aligned to tactile_hand_side)

Usage (after extraction):
  python scripts/plot_latent_tsne.py \
      --root eval_artifacts/latent_tsne \
      --out  eval_artifacts/latent_tsne/figures

Smoke test (no data needed):
  python scripts/plot_latent_tsne.py --synthetic --out /tmp/tsne_smoke
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless; no display on a login node
import matplotlib.pyplot as plt

# Default SDH fingertip ordering. Override with --finger-names if the
# corpus uses a different index convention.
DEFAULT_FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
# Hand slot indices are not anatomy, and for this corpus the obvious guess is
# backwards. Three of the six tasks are single-handed and right-only, so slot 0
# is the hand every task records -- the right one -- and slot 1 is the left,
# which only the bimanual tasks contribute. Override with --hand-names for a
# corpus that orders them differently.
HAND_NAMES = {0: "right", 1: "left", 2: "hand2", 3: "hand3"}


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------
def _load_task(task_dir: str) -> Optional[Dict]:
    """Load one task's latents.npz into a dict."""
    npz_path = os.path.join(task_dir, "latents.npz")
    if not os.path.isfile(npz_path):
        return None
    d = dict(np.load(npz_path, allow_pickle=True))
    task_name = d.get("task")
    if task_name is None:
        task_name = os.path.basename(task_dir.rstrip("/"))
    else:
        task_name = str(task_name.item() if hasattr(task_name, "item") else task_name)
    d["task"] = task_name
    d["_dir"] = task_dir
    return d


def _discover_tasks(root: str) -> List[Dict]:
    tasks = []
    for task_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(task_dir):
            continue
        loaded = _load_task(task_dir)
        if loaded is not None:
            tasks.append(loaded)
    return tasks


def _hand_name(side: int) -> str:
    return HAND_NAMES.get(int(side), f"hand{int(side)}")


# ----------------------------------------------------------------------
# Embedding + metric helpers
# ----------------------------------------------------------------------
def _fit_tsne(X: np.ndarray, seed: int, pca_dim: int = 50,
              perplexity: float = 30.0) -> np.ndarray:
    """PCA(pca_dim) -> t-SNE(2). Robust to small N (perplexity clamps)."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 3:
        if X.shape[1] >= 2 and n >= 2:
            return PCA(n_components=2).fit_transform(X)
        return np.zeros((n, 2))
    d = min(pca_dim, X.shape[1], n - 1)
    if d >= 2 and X.shape[1] > d:
        X = PCA(n_components=d, random_state=seed).fit_transform(X)
    perp = float(max(5.0, min(perplexity, (n - 1) / 3.0)))
    tsne = TSNE(n_components=2, perplexity=perp, init="pca",
                random_state=seed, max_iter=1000)
    return tsne.fit_transform(X)


def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette in whatever space X lives in; NaN if <2 classes."""
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2 or len(labels) < 3:
        return float("nan")
    try:
        return float(silhouette_score(np.asarray(X, dtype=np.float64), labels))
    except Exception:
        return float("nan")


def _classify_128d(X: np.ndarray, y: np.ndarray, seed: int,
                   knn_k: int = 5, n_splits: int = 5) -> Dict:
    """Raw-latent-space task-separability metrics.

    All classifiers run on the ORIGINAL high-dim latent (no t-SNE). Each CV
    fold fits its own StandardScaler on the TRAIN fold only (leakage-free).

    * kNN: fixed k (default 5), stratified k-fold.
    * linear probe: multinomial logistic regression (sklearn>=1.5 uses the
      multinomial formulation by default for multiclass with lbfgs, so we do
      not pass the deprecated multi_class arg).

    CV is sample-level stratified (the current .npz has no episode id). This
    is recorded in the returned dict as `cv_scheme`; if same-episode
    consecutive samples exist, sample-level CV can overestimate accuracy via
    temporal correlation -> switch to GroupKFold by episode once episode_id
    is extracted.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    out: Dict = {
        "n": int(len(y)),
        "n_classes": int(len(classes)),
        "chance": (1.0 / len(classes)) if len(classes) > 0 else float("nan"),
        "cv_scheme": f"sample-level stratified {n_splits}-fold",
        "knn_k": int(knn_k),
        "silhouette_128d": _silhouette(X, y),
    }
    min_count = int(counts.min()) if len(counts) else 0
    n_splits_eff = min(n_splits, min_count)
    if len(classes) < 2 or n_splits_eff < 2:
        out.update({
            "n_splits": int(max(0, n_splits_eff)),
            "knn_accuracy_mean": float("nan"), "knn_accuracy_std": float("nan"),
            "linear_probe_accuracy_mean": float("nan"),
            "linear_probe_accuracy_std": float("nan"),
        })
        return out

    out["n_splits"] = int(n_splits_eff)
    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)
    # Guard k against the smallest train-fold class size.
    k_eff = int(max(1, min(knn_k, min_count - 1)))
    out["knn_k"] = k_eff

    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k_eff))
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=1000))  # multinomial by default

    knn_scores = cross_val_score(knn, X, y, cv=skf)
    lr_scores = cross_val_score(lr, X, y, cv=skf)
    out.update({
        "knn_accuracy_mean": float(knn_scores.mean()),
        "knn_accuracy_std": float(knn_scores.std()),
        "linear_probe_accuracy_mean": float(lr_scores.mean()),
        "linear_probe_accuracy_std": float(lr_scores.std()),
    })
    return out


def _bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0,
                  alpha: float = 0.05) -> tuple:
    """Percentile bootstrap CI for the mean. Returns (lo, hi)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return (0.0, 0.0)
    if n < 2:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = x[idx].mean(axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def _task_palette(task_names: List[str]) -> Dict[str, tuple]:
    cmap = plt.get_cmap("tab10")
    return {t: cmap(i % 10) for i, t in enumerate(task_names)}


def _all_hand_sides(tasks: List[Dict]) -> List[int]:
    sides = set()
    for t in tasks:
        s = t.get("tactile_hand_side")
        if s is not None and len(s) > 0:
            sides.update(int(v) for v in np.unique(np.asarray(s)))
    return sorted(sides)


# ----------------------------------------------------------------------
# Claim 1: task specificity, per hand (Fig 1L / Fig 1R)
# ----------------------------------------------------------------------
def fig_task_tsne_per_hand(tasks: List[Dict], out_dir: str, seed: int,
                           palette: Dict[str, tuple]) -> Dict:
    """Per-hand fused-latent t-SNE by task + raw 128-D separability metrics."""
    result: Dict = {}
    figures: Dict = {}
    for side in _all_hand_sides(tasks):
        hand = _hand_name(side)
        Xs, task_lab = [], []
        for t in tasks:
            h = t.get("tactile_hand")
            hs = t.get("tactile_hand_side")
            if h is None or hs is None or len(h) == 0:
                continue
            m = np.asarray(hs).astype(int) == side
            if not m.any():
                continue
            Xs.append(np.asarray(h, dtype=np.float64)[m])
            task_lab += [t["task"]] * int(m.sum())
        if not Xs:
            continue
        X = np.concatenate(Xs, axis=0)
        task_lab = np.array(task_lab)

        # Headline evidence: raw 128-D metrics.
        metrics = _classify_128d(X, task_lab, seed=seed)

        # Visualization only.
        X2 = _fit_tsne(X, seed=seed)
        metrics["tsne2d_silhouette_diagnostic"] = _silhouette(X2, task_lab)
        result[hand] = metrics

        fig, ax = plt.subplots(figsize=(8, 7))
        for task in sorted(set(task_lab)):
            mm = task_lab == task
            ax.scatter(X2[mm, 0], X2[mm, 1], s=22, alpha=0.7,
                       color=palette[task], edgecolors="none", label=task)
        ax.set_title(
            f"Fig 1 ({hand} hand): fused tactile latent t-SNE by task\n"
            f"128-D: silhouette={metrics['silhouette_128d']:.3f}  "
            f"kNN(k={metrics['knn_k']})={metrics['knn_accuracy_mean']:.3f}"
            f"\u00b1{metrics['knn_accuracy_std']:.3f}  "
            f"logit={metrics['linear_probe_accuracy_mean']:.3f}"
            f"\u00b1{metrics['linear_probe_accuracy_std']:.3f}  "
            f"(chance={metrics['chance']:.2f})",
            fontsize=10,
        )
        ax.set_xlabel("t-SNE 1 (viz only)"); ax.set_ylabel("t-SNE 2 (viz only)")
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
        fig.tight_layout()
        path = os.path.join(out_dir, f"fig1_task_tsne_{hand}.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        figures[hand] = path
        print(f"[fig1:{hand}] wrote {path}  "
              f"silhouette_128d={metrics['silhouette_128d']:.3f} "
              f"kNN={metrics['knn_accuracy_mean']:.3f} "
              f"logit={metrics['linear_probe_accuracy_mean']:.3f} "
              f"chance={metrics['chance']:.2f}")
    return {"task_specificity": result, "_fig1_paths": figures}


# ----------------------------------------------------------------------
# Comparison: RGB (frozen LTX VAE) latent t-SNE by task
# ----------------------------------------------------------------------
def fig_vision_task_tsne(tasks: List[Dict], out_dir: str, seed: int,
                         palette: Dict[str, tuple]) -> Dict:
    """RGB head-view latent t-SNE by task + raw 128-D metrics.

    Same frozen LTX VAE that encodes tactile (via GrayToRGB). This is a
    side-by-side COMPARISON for Fig 1 (the tactile task t-SNE): it shows the
    RGB latent is also task-specific. It makes NO cross-modal alignment claim
    -- the two modalities are analyzed separately, each in its own space.
    """
    Xs, task_lab = [], []
    for t in tasks:
        v = t.get("vision")
        if v is None or len(v) == 0:
            continue
        Xs.append(np.asarray(v, dtype=np.float64))
        task_lab += [t["task"]] * len(v)
    if not Xs:
        print("[figV] no vision latents; skipping.")
        return {"vision_task_specificity": {}, "_figV_path": None}
    X = np.concatenate(Xs, axis=0)
    task_lab = np.array(task_lab)

    metrics = _classify_128d(X, task_lab, seed=seed)
    X2 = _fit_tsne(X, seed=seed)
    metrics["tsne2d_silhouette_diagnostic"] = _silhouette(X2, task_lab)

    fig, ax = plt.subplots(figsize=(8, 7))
    for task in sorted(set(task_lab)):
        mm = task_lab == task
        ax.scatter(X2[mm, 0], X2[mm, 1], s=22, alpha=0.7,
                   color=palette[task], edgecolors="none", label=task)
    ax.set_title(
        "Fig V (comparison): RGB head-view latent t-SNE by task "
        "(frozen LTX VAE)\n"
        f"128-D: silhouette={metrics['silhouette_128d']:.3f}  "
        f"kNN(k={metrics['knn_k']})={metrics['knn_accuracy_mean']:.3f}"
        f"\u00b1{metrics['knn_accuracy_std']:.3f}  "
        f"logit={metrics['linear_probe_accuracy_mean']:.3f}"
        f"\u00b1{metrics['linear_probe_accuracy_std']:.3f}  "
        f"(chance={metrics['chance']:.2f})",
        fontsize=10,
    )
    ax.set_xlabel("t-SNE 1 (viz only)"); ax.set_ylabel("t-SNE 2 (viz only)")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(out_dir, "figV_vision_task_tsne.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[figV] wrote {path}  silhouette_128d={metrics['silhouette_128d']:.3f} "
          f"kNN={metrics['knn_accuracy_mean']:.3f} "
          f"logit={metrics['linear_probe_accuracy_mean']:.3f} "
          f"chance={metrics['chance']:.2f}")
    return {"vision_task_specificity": metrics, "_figV_path": path}


# ----------------------------------------------------------------------
# Claim 2: per-finger sensitivity retained, per hand (Fig 2L / Fig 2R)
# ----------------------------------------------------------------------
def fig_fusion_shift_per_hand(tasks: List[Dict], out_dir: str,
                              finger_names: List[str], seed: int,
                              shift_key: str = "fusion_shift",
                              suffix: str = "") -> Dict:
    """Per-hand leave-one-finger-out fusion shift, mean + 95% bootstrap CI."""
    result: Dict = {}
    figures: Dict = {}
    # Determine F from the first task that has the shift array.
    F = None
    for t in tasks:
        s = t.get(shift_key)
        if s is not None and len(s) > 0:
            F = np.asarray(s).shape[1]
            break
    if F is None:
        print(f"[fig2{suffix}] no {shift_key} data; skipping.")
        return {"per_finger_sensitivity": {}, "_fig2_paths": {}}
    names = [finger_names[i] if i < len(finger_names) else f"F{i}"
             for i in range(F)]

    for side in _all_hand_sides(tasks):
        hand = _hand_name(side)
        rows = []
        for t in tasks:
            s = t.get(shift_key)
            hs = t.get("tactile_hand_side")
            if s is None or hs is None or len(s) == 0:
                continue
            m = np.asarray(hs).astype(int) == side
            if not m.any():
                continue
            rows.append(np.asarray(s, dtype=np.float64)[m])
        if not rows:
            continue
        shift = np.concatenate(rows, axis=0)  # (N_side, F)

        means = shift.mean(0)
        cis = [_bootstrap_ci(shift[:, i], seed=seed + i) for i in range(F)]
        lo = np.array([c[0] for c in cis])
        hi = np.array([c[1] for c in cis])
        yerr = np.vstack([means - lo, hi - means])

        result[hand] = {
            "n": int(shift.shape[0]),
            "per_finger": {
                names[i]: {
                    "mean_shift": float(means[i]),
                    "ci95_lo": float(lo[i]),
                    "ci95_hi": float(hi[i]),
                }
                for i in range(F)
            },
            "min_finger_mean_shift": float(means.min()),
            "all_fingers_ci_above_zero": bool((lo > 0).all()),
        }

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(np.arange(F), means, yerr=yerr, capsize=5,
               color=plt.get_cmap("tab10")(np.arange(F) % 10))
        ax.set_xticks(np.arange(F)); ax.set_xticklabels(names)
        ax.set_ylabel(r"normalized shift  $\|z_{all}-z_{-i}\|/\|z_{all}\|$")
        abl = "mean-token" if "mean" in suffix else "zero-token"
        ax.set_title(f"Fig 2 ({hand} hand): leave-one-finger-out fusion shift "
                     f"[{abl}] (N={shift.shape[0]}, 95% bootstrap CI)",
                     fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"fig2_fusion_shift{suffix}_{hand}.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        figures[hand] = path
        print(f"[fig2{suffix}:{hand}] wrote {path}  "
              f"mean/finger={np.round(means, 4).tolist()}  "
              f"all_ci>0={result[hand]['all_fingers_ci_above_zero']}")
    return {"per_finger_sensitivity": result, "_fig2_paths": figures}


def fig_fusion_shift_per_task(tasks: List[Dict], out_dir: str,
                              finger_names: List[str], seed: int,
                              shift_key: str = "fusion_shift",
                              suffix: str = "") -> Dict:
    """Leave-one-finger-out fusion shift computed *per task*, hands separate.

    Same metric as `fig_fusion_shift_per_hand` (‖z_all − z_−i‖/‖z_all‖ with 95%
    bootstrap CI), but not pooled over tasks: for every (task, hand) pair we
    report the per-finger shift. Emits one grouped-bar figure per hand
    (x = finger, one bar group per task) and a nested summary dict.
    """
    result: Dict = {}
    figures: Dict = {}
    F = None
    for t in tasks:
        s = t.get(shift_key)
        if s is not None and len(s) > 0:
            F = np.asarray(s).shape[1]
            break
    if F is None:
        print(f"[fig2b{suffix}] no {shift_key} data; skipping.")
        return {"per_task_finger_sensitivity": {}, "_fig2b_paths": {}}
    names = [finger_names[i] if i < len(finger_names) else f"F{i}"
             for i in range(F)]
    task_names = sorted({t["task"] for t in tasks})

    for side in _all_hand_sides(tasks):
        hand = _hand_name(side)
        per_task_means: Dict[str, np.ndarray] = {}
        per_task_ci: Dict[str, tuple] = {}
        for t in tasks:
            s = t.get(shift_key)
            hs = t.get("tactile_hand_side")
            if s is None or hs is None or len(s) == 0:
                continue
            m = np.asarray(hs).astype(int) == side
            if not m.any():
                continue
            shift = np.asarray(s, dtype=np.float64)[m]  # (N, F)
            means = shift.mean(0)
            cis = [_bootstrap_ci(shift[:, i], seed=seed + i) for i in range(F)]
            lo = np.array([c[0] for c in cis])
            hi = np.array([c[1] for c in cis])
            tname = t["task"]
            per_task_means[tname] = means
            per_task_ci[tname] = (lo, hi)
            result.setdefault(tname, {})[hand] = {
                "n": int(shift.shape[0]),
                "per_finger": {
                    names[i]: {
                        "mean_shift": float(means[i]),
                        "ci95_lo": float(lo[i]),
                        "ci95_hi": float(hi[i]),
                    }
                    for i in range(F)
                },
                "all_fingers_ci_above_zero": bool((lo > 0).all()),
            }
        if not per_task_means:
            continue

        present = [tn for tn in task_names if tn in per_task_means]
        T = len(present)
        width = 0.8 / max(T, 1)
        x = np.arange(F)
        cmap = plt.get_cmap("tab10")
        fig, ax = plt.subplots(figsize=(max(8.0, 1.6 * F), 5))
        for j, tn in enumerate(present):
            means = per_task_means[tn]
            lo, hi = per_task_ci[tn]
            yerr = np.vstack([means - lo, hi - means])
            ax.bar(x + (j - (T - 1) / 2.0) * width, means, width=width,
                   yerr=yerr, capsize=3, color=cmap(j % 10), label=tn)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        abl = "mean-token" if "mean" in suffix else "zero-token"
        ax.set_ylabel(r"normalized shift  $\|z_{all}-z_{-i}\|/\|z_{all}\|$")
        ax.set_title(f"Fig 2b ({hand} hand): leave-one-finger-out fusion shift "
                     f"per task [{abl}] (95% bootstrap CI)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(
            out_dir, f"fig2b_fusion_shift_per_task{suffix}_{hand}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures[hand] = path
        print(f"[fig2b{suffix}:{hand}] wrote {path}")
        for tn in present:
            print(f"    {tn:>28s}: {np.round(per_task_means[tn], 4).tolist()}")
    return {"per_task_finger_sensitivity": result, "_fig2b_paths": figures}


# ----------------------------------------------------------------------
# Supplementary: pre-fusion per-finger t-SNE, per hand (Fig 3L / Fig 3R)
# ----------------------------------------------------------------------
def fig_finger_tsne_per_hand(tasks: List[Dict], out_dir: str, seed: int,
                             finger_names: List[str],
                             facet_by_task: bool = False) -> Dict:
    """Per-hand pre-fusion per-finger t-SNE, all tasks pooled (supplementary)."""
    result: Dict = {}
    figures: Dict = {}
    for side in _all_hand_sides(tasks):
        hand = _hand_name(side)
        Xs, fid_lab, task_lab = [], [], []
        for t in tasks:
            xf = t.get("tactile_finger")
            fs = t.get("finger_side")
            fi = t.get("finger_id")
            if xf is None or fs is None or fi is None or len(xf) == 0:
                continue
            m = np.asarray(fs).astype(int) == side
            if not m.any():
                continue
            Xs.append(np.asarray(xf, dtype=np.float64)[m])
            fid_lab.append(np.asarray(fi).astype(int)[m])
            task_lab += [t["task"]] * int(m.sum())
        if not Xs:
            continue
        X = np.concatenate(Xs, axis=0)
        fid = np.concatenate(fid_lab, axis=0)
        task_lab = np.array(task_lab)

        X2 = _fit_tsne(X, seed=seed)
        sil_128 = _silhouette(X, fid)
        result[hand] = {
            "finger_silhouette_128d": sil_128,
            "finger_silhouette_tsne2d_diagnostic": _silhouette(X2, fid),
            "n": int(len(fid)),
        }

        cmap = plt.get_cmap("tab10")
        fig, ax = plt.subplots(figsize=(8, 7))
        for f in sorted(set(fid)):
            mm = fid == f
            name = finger_names[f] if f < len(finger_names) else f"F{f}"
            ax.scatter(X2[mm, 0], X2[mm, 1], s=14, alpha=0.7,
                       color=cmap(f % 10), edgecolors="none", label=name)
        ax.set_title(f"Fig 3 ({hand} hand, supplementary): pre-fusion "
                     f"per-finger t-SNE (all tasks pooled)\n"
                     f"finger silhouette 128-D={sil_128:.3f}", fontsize=10)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, f"fig3_finger_tsne_{hand}.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        figures[hand] = path
        print(f"[fig3:{hand}] wrote {path}  finger_silhouette_128d={sil_128:.3f}")

        if facet_by_task:
            uniq = sorted(set(task_lab))
            ncol = 2 if len(uniq) > 1 else 1
            nrow = int(np.ceil(len(uniq) / ncol))
            figt, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5 * nrow),
                                      squeeze=False)
            for i, task in enumerate(uniq):
                axt = axes[i // ncol][i % ncol]
                tm = task_lab == task
                X2t = _fit_tsne(X[tm], seed=seed)
                for f in sorted(set(fid[tm])):
                    fm = fid[tm] == f
                    name = finger_names[f] if f < len(finger_names) else f"F{f}"
                    axt.scatter(X2t[fm, 0], X2t[fm, 1], s=14, alpha=0.7,
                                color=cmap(f % 10), edgecolors="none", label=name)
                axt.set_title(f"{task}", fontsize=9)
                axt.legend(loc="best", fontsize=6)
            for j in range(len(uniq), nrow * ncol):
                axes[j // ncol][j % ncol].axis("off")
            figt.suptitle(f"Fig 3 ({hand}) supplementary: per-finger t-SNE "
                          f"faceted by task", fontsize=12)
            figt.tight_layout(rect=[0, 0, 1, 0.97])
            pfac = os.path.join(out_dir, f"fig3_finger_tsne_{hand}_by_task.png")
            figt.savefig(pfac, dpi=150); plt.close(figt)
            figures[f"{hand}_by_task"] = pfac
            print(f"[fig3:{hand}] wrote faceted {pfac}")
    return {"supplementary_per_finger_tsne": result, "_fig3_paths": figures}


# ----------------------------------------------------------------------
# Diagnostic (NOT a paper figure): vision vs tactile modality separation
# ----------------------------------------------------------------------
def diagnostic_vision_tactile_separation(tasks: List[Dict], out_dir: str,
                                         seed: int, emit_fig: bool) -> Dict:
    """Modality-SEPARATION metric in raw 128-D space (higher = more separated).

    Deliberately NOT framed as overlap/alignment/shared-manifold. Renders an
    overlaid t-SNE only when emit_fig=True, purely for internal inspection.
    """
    per_task: Dict = {}
    for t in tasks:
        v = t.get("vision"); h = t.get("tactile_hand")
        if v is None or h is None or len(v) == 0 or len(h) == 0:
            continue
        v = np.asarray(v, dtype=np.float64); h = np.asarray(h, dtype=np.float64)
        cv, ch = v.mean(0), h.mean(0)
        centroid_dist = float(np.linalg.norm(cv - ch))
        intra = float(0.5 * (v.std(0).mean() + h.std(0).mean()) + 1e-9)
        per_task[t["task"]] = float(centroid_dist / intra)

    diag = {
        "metric": "centroid_distance_over_intra_modal_spread",
        "space": "raw_128d",
        "interpretation": "Higher values indicate stronger modality separation "
                          "(NOT overlap).",
        "claim": "diagnostic_only",
        "per_task": per_task,
        "mean_ratio": float(np.mean(list(per_task.values()))) if per_task else float("nan"),
    }

    if emit_fig and per_task:
        Xs, task_lab, modality = [], [], []
        for t in tasks:
            v = t.get("vision"); h = t.get("tactile_hand")
            if v is not None and len(v) > 0:
                Xs.append(np.asarray(v, dtype=np.float64))
                task_lab += [t["task"]] * len(v); modality += [0] * len(v)
            if h is not None and len(h) > 0:
                Xs.append(np.asarray(h, dtype=np.float64))
                task_lab += [t["task"]] * len(h); modality += [1] * len(h)
        X = np.concatenate(Xs, axis=0)
        task_lab = np.array(task_lab); modality = np.array(modality)
        X2 = _fit_tsne(X, seed=seed)
        palette = _task_palette(sorted(set(task_lab)))
        fig, ax = plt.subplots(figsize=(10, 8))
        for task in sorted(set(task_lab)):
            for mod, marker, mname in [(0, "o", "RGB"), (1, "^", "tactile")]:
                m = (task_lab == task) & (modality == mod)
                if not m.any():
                    continue
                ax.scatter(X2[m, 0], X2[m, 1], s=20, alpha=0.6,
                           color=palette[task], marker=marker,
                           edgecolors="none", label=f"{task} [{mname}]")
        ax.set_title("DIAGNOSTIC (not a paper figure): vision (o) vs tactile "
                     "(^) modality separation\n"
                     "distinct modality-specific regions; no alignment claim",
                     fontsize=10)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.legend(loc="best", fontsize=7, framealpha=0.9)
        fig.tight_layout()
        path = os.path.join(out_dir, "diagnostic_vision_tactile_separation.png")
        fig.savefig(path, dpi=150); plt.close(fig)
        diag["figure_path"] = path
        print(f"[diagnostic] wrote {path}  mean_ratio={diag['mean_ratio']:.2f}")
    else:
        print(f"[diagnostic] vision_tactile_separation mean_ratio="
              f"{diag['mean_ratio']:.2f} (figure not emitted)")
    return {"diagnostics": {"vision_tactile_separation": diag}}


# ----------------------------------------------------------------------
# Synthetic data (smoke test)
# ----------------------------------------------------------------------
def _make_synthetic(root: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    D, F = 128, 5
    task_defs = ["erase", "chip", "cube_one_hand", "cube_handover"]
    for ti, task in enumerate(task_defs):
        tdir = os.path.join(root, task)
        os.makedirs(tdir, exist_ok=True)
        N = 60
        center = rng.normal(scale=3.0, size=D) * (ti + 1)
        tactile_hand = center + rng.normal(scale=1.0, size=(2 * N, D))
        hand_side = np.array([0] * N + [1] * N)
        vision = center * 0.7 + rng.normal(scale=1.0, size=(N, D)) + 5.0
        fcenters = center[None, :] + rng.normal(scale=2.0, size=(F, D))
        tf, fid, fside = [], [], []
        for h in range(2):
            for f in range(F):
                tf.append(fcenters[f] + rng.normal(scale=0.6, size=(N, D)))
                fid += [f] * N; fside += [h] * N
        tactile_finger = np.concatenate(tf, axis=0)
        base = np.array([0.25, 0.22, 0.18, 0.15, 0.12])
        fusion_shift = np.abs(base[None, :] + rng.normal(scale=0.03, size=(2 * N, F)))
        # mean-token ablation is a milder perturbation than zeroing:
        fusion_shift_mean = np.abs(0.6 * base[None, :]
                                   + rng.normal(scale=0.03, size=(2 * N, F)))
        np.savez(
            os.path.join(tdir, "latents.npz"),
            task=np.array(task),
            vision=vision.astype(np.float32),
            tactile_hand=tactile_hand.astype(np.float32),
            tactile_hand_side=hand_side.astype(np.int64),
            tactile_finger=tactile_finger.astype(np.float32),
            finger_id=np.array(fid, dtype=np.int64),
            finger_side=np.array(fside, dtype=np.int64),
            fusion_shift=fusion_shift.astype(np.float32),
            fusion_shift_mean=fusion_shift_mean.astype(np.float32),
        )
    print(f"[synthetic] wrote {len(task_defs)} synthetic tasks under {root}")


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="eval_artifacts/latent_tsne",
                    help="Dir containing per-task subdirs with latents.npz")
    ap.add_argument("--out", default=None,
                    help="Output dir for figures (default <root>/figures)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--finger-names", nargs="+", default=DEFAULT_FINGER_NAMES)
    ap.add_argument("--hand-names", nargs="+", default=None,
                    help="Anatomical names for hand slots 0,1,... Defaults to "
                         "'right left', which is this corpus's ordering; see "
                         "HAND_NAMES. Figure filenames use these names.")
    ap.add_argument("--facet-by-task", action="store_true",
                    help="Also emit the per-hand per-finger t-SNE faceted by "
                         "task (supplementary diagnostic; default OFF).")
    ap.add_argument("--emit-diagnostic-separation", action="store_true",
                    help="Also render the vision-vs-tactile modality-separation "
                         "t-SNE PNG (internal diagnostic; not a paper figure). "
                         "The separation metric is always written to summary.json.")
    ap.add_argument("--synthetic", action="store_true",
                    help="Generate synthetic data under --root and plot (smoke).")
    args = ap.parse_args()

    if args.hand_names:
        HAND_NAMES.update({i: n for i, n in enumerate(args.hand_names)})

    out_dir = args.out or os.path.join(args.root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    if args.synthetic:
        _make_synthetic(args.root, seed=args.seed)

    tasks = _discover_tasks(args.root)
    if not tasks:
        print(f"[plot] no task .npz found under {args.root}. "
              f"Run extract_latents_for_tsne.py first (or --synthetic).")
        return 1
    print(f"[plot] loaded {len(tasks)} tasks: {[t['task'] for t in tasks]}")

    palette = _task_palette(sorted({t["task"] for t in tasks}))
    hands = [_hand_name(s) for s in _all_hand_sides(tasks)]

    summary: Dict = {"tasks": [t["task"] for t in tasks], "hands": hands}
    figures: Dict = {}

    r1 = fig_task_tsne_per_hand(tasks, out_dir, args.seed, palette)
    summary["task_specificity"] = r1["task_specificity"]
    figures["fig1_task_tsne"] = r1["_fig1_paths"]

    rv = fig_vision_task_tsne(tasks, out_dir, args.seed, palette)
    summary["vision_task_specificity"] = rv["vision_task_specificity"]
    figures["figV_vision_task_tsne"] = rv["_figV_path"]

    r2 = fig_fusion_shift_per_hand(tasks, out_dir, args.finger_names, args.seed)
    summary["per_finger_sensitivity"] = r2["per_finger_sensitivity"]
    figures["fig2_fusion_shift"] = r2["_fig2_paths"]

    r2b = fig_fusion_shift_per_task(tasks, out_dir, args.finger_names, args.seed)
    summary["per_task_finger_sensitivity"] = r2b["per_task_finger_sensitivity"]
    figures["fig2b_fusion_shift_per_task"] = r2b["_fig2b_paths"]

    # Mean-token ablation variant (emitted only if extraction stored it).
    if any(t.get("fusion_shift_mean") is not None for t in tasks):
        r2m = fig_fusion_shift_per_hand(
            tasks, out_dir, args.finger_names, args.seed,
            shift_key="fusion_shift_mean", suffix="_mean")
        summary["per_finger_sensitivity_mean"] = r2m["per_finger_sensitivity"]
        figures["fig2_fusion_shift_mean"] = r2m["_fig2_paths"]

        r2bm = fig_fusion_shift_per_task(
            tasks, out_dir, args.finger_names, args.seed,
            shift_key="fusion_shift_mean", suffix="_mean")
        summary["per_task_finger_sensitivity_mean"] = \
            r2bm["per_task_finger_sensitivity"]
        figures["fig2b_fusion_shift_per_task_mean"] = r2bm["_fig2b_paths"]

    r3 = fig_finger_tsne_per_hand(tasks, out_dir, args.seed, args.finger_names,
                                  facet_by_task=args.facet_by_task)
    summary["supplementary_per_finger_tsne"] = r3["supplementary_per_finger_tsne"]
    figures["fig3_finger_tsne"] = r3["_fig3_paths"]

    rd = diagnostic_vision_tactile_separation(
        tasks, out_dir, args.seed, emit_fig=args.emit_diagnostic_separation)
    summary["diagnostics"] = rd["diagnostics"]

    summary["figures"] = figures

    spath = os.path.join(out_dir, "summary.json")
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[plot] wrote {spath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
