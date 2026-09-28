#!/usr/bin/env python3
"""scripts/plot_prefusion_vs_adapter_tsne.py

Tactile representation BEFORE vs AFTER the multi-finger adapter, coloured by
task, over the paper's six evaluation tasks. Reads the per-task `latents.npz`
written by extract_latents_for_tsne.py; no model, no GPU.

The two panels support two separate claims:

  Panel 1  single finger, frozen vision VAE
      One point per sample: the frozen visual-VAE latent of ONE fingertip.
      Claim: a VAE pretrained only on natural video already extracts
      task-discriminative structure from a single tactile fingertip, with no
      tactile-specific encoder trained for it.

  Panel 2  whole hand, adapter fused
      One point per sample: the FingerSetTransformerAdapter output that the
      DiT actually consumes, compressing five fingertips into one hand token.
      Claim: that 5 -> 1 compression PRESERVES the task-discriminative
      structure rather than washing it out.

WHY THE THUMB AND WHY THE RIGHT HAND. The panels must be comparable across all
six tasks, so both are restricted to hardware every task exercises. Three of
the six tasks are single-handed and use the right hand only, so the right hand
is the sole hand common to all six; the left hand appears only in the bimanual
tasks. Within that hand the thumb is the fingertip every task loads (and it
carries the largest leave-one-finger-out fusion shift), so it is the honest
single-finger representative.

HAND SLOT INDICES ARE NOT ANATOMY. `tactile_hand_side` is a slot index. In this
corpus slot 0 is the RIGHT hand for every task (it is the only hand the
single-handed tasks record) and slot 1 is the left hand, present only in the
bimanual tasks. Note this is the opposite of `HAND_NAMES` in
plot_latent_tsne.py, which assumes 0=left; pass --hand-slot/--hand-name here
rather than trusting that mapping.

FAIRNESS. Both panels carry one point per (sample, hand) over the same tasks,
the same hand and the same underlying frames, so point count, class balance and
chance level are identical and the only thing that varies is the
representation. The optional --with-meanpool panel adds a parameter-free
5 -> 1 average as a fusion baseline, isolating what the LEARNED fusion adds
over naive pooling.

Usage:
  python scripts/plot_prefusion_vs_adapter_tsne.py \
      --root eval_artifacts/latent_tsne \
      --out  eval_artifacts/latent_tsne/figures_prefusion
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# Corpus tags are snake_case; the paper names the tasks like this, and a legend
# is the one place a reader maps colors onto the task table.
PRETTY = {
    "bottle_cap": "Bottle Cap",
    "bowl": "Bowl",
    "cube_handover": "Cube Handover",
    "cube_place": "Cube Place",
    "tongs": "Tongs",
    "wipe": "Two-Hand Wipe",
}


def _metrics(X: np.ndarray, y: np.ndarray, seed: int, njobs: int = 4) -> dict:
    """Task discriminability in the RAW latent space (t-SNE is viz only).

    njobs is deliberately small and never -1: on a 128-core shared login node
    -1 fans out to every core and the job gets reaped by the node's resource
    policy, which looks exactly like a silent hang.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    classes = np.unique(y)
    out = {
        "n": int(len(y)),
        "n_classes": int(len(classes)),
        "chance": float(1.0 / len(classes)),
    }
    print("      silhouette ...", end="", flush=True)
    out["silhouette_128d"] = float(silhouette_score(X, y))
    print(f" {out['silhouette_128d']:+.3f}", flush=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    print("      kNN ...", end="", flush=True)
    knn = cross_val_score(KNeighborsClassifier(n_neighbors=5), X, y,
                          cv=cv, n_jobs=njobs)
    out["knn_accuracy_mean"] = float(knn.mean())
    out["knn_accuracy_std"] = float(knn.std())
    print(f" {knn.mean():.3f}", flush=True)

    print("      probe ...", end="", flush=True)
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    lp = cross_val_score(probe, X, y, cv=cv, n_jobs=njobs)
    out["linear_probe_accuracy_mean"] = float(lp.mean())
    out["linear_probe_accuracy_std"] = float(lp.std())
    print(f" {lp.mean():.3f}", flush=True)
    return out


def _tsne(X: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    Z = X
    if Z.shape[1] > 50:
        Z = PCA(n_components=50, random_state=seed).fit_transform(Z)
    perp = float(min(30, max(5, (len(Z) - 1) // 3)))
    return TSNE(n_components=2, perplexity=perp, init="pca",
                random_state=seed, learning_rate="auto").fit_transform(Z)


def _load(root: str) -> dict:
    per_task = {}
    for p in sorted(glob.glob(os.path.join(root, "*", "latents.npz"))):
        per_task[os.path.basename(os.path.dirname(p))] = np.load(p, allow_pickle=True)
    if not per_task:
        raise SystemExit(f"no latents.npz under {root}")
    return per_task


def _build(per_task: dict, slot: int, finger: int, with_meanpool: bool) -> dict:
    """One point per (sample, hand) for each representation, same rows."""
    fing_X, pool_X, adap_X, lab = [], [], [], []

    for tag, d in per_task.items():
        m_hand = d["tactile_hand_side"] == slot
        if not m_hand.any():
            print(f"   [skip] {tag}: no hand slot {slot}")
            continue

        adap = d["tactile_hand"][m_hand]

        # tactile_finger rows are ordered (sample, hand, finger) with finger
        # fastest, so this reshape recovers the hand rows that tactile_hand
        # is aligned to, and axis 1 indexes the fingertip.
        F = int(d["finger_id"].max()) + 1
        allf = d["tactile_finger"].reshape(-1, F, d["tactile_finger"].shape[1])
        side_rows = d["finger_side"].reshape(-1, F)[:, 0]
        sel = allf[side_rows == slot]                 # (n_hand_rows, F, C)
        assert len(sel) == len(adap), (
            f"{tag}: {len(sel)} finger rows vs {len(adap)} adapter rows")

        fing_X.append(sel[:, finger])                 # ONE fingertip
        adap_X.append(adap)
        if with_meanpool:
            pool_X.append(sel.mean(axis=1))
        lab += [tag] * len(adap)

    y = np.array(lab)
    reps = {
        f"{FINGER_NAMES[finger]} only \u2014 frozen vision VAE":
            (np.concatenate(fing_X), y),
    }
    if with_meanpool:
        reps["5 fingers \u2014 mean pool (naive)"] = (np.concatenate(pool_X), y)
    reps["5 fingers \u2192 1 hand \u2014 tactile compressor"] = (np.concatenate(adap_X), y)
    return reps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="eval_artifacts/latent_tsne")
    ap.add_argument("--out", default="eval_artifacts/latent_tsne/figures_prefusion")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hand-slot", type=int, default=0,
                    help="Slot index; 0 is the right hand in this corpus.")
    ap.add_argument("--hand-name", default="right",
                    help="Anatomical name for --hand-slot, used in titles.")
    ap.add_argument("--finger", type=int, default=0,
                    help="Fingertip index for the single-finger panel "
                         "(0=thumb, the one every task loads).")
    ap.add_argument("--ext", default="png",
                    help="Figure format, e.g. pdf for a vector figure to embed "
                         "in the paper.")
    ap.add_argument("--njobs", type=int, default=4,
                    help="Cross-validation parallelism. Keep small on shared "
                         "login nodes; -1 gets the job reaped.")
    ap.add_argument("--with-meanpool", action="store_true",
                    help="Add a parameter-free 5->1 mean-pool baseline panel.")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    # IEEE PDF eXpress rejects Type 3 fonts; 42 embeds TrueType instead.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    per_task = _load(args.root)
    reps = _build(per_task, args.hand_slot, args.finger, args.with_meanpool)

    ncol = len(reps)
    fig, axes = plt.subplots(1, ncol, figsize=(6.2 * ncol, 5.6))
    if ncol == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")
    colour = {t: cmap(i % 10) for i, t in enumerate(sorted(per_task.keys()))}
    summary = {}

    for ax, (name, (X, y)) in zip(axes, reps.items()):
        print(f"   [{name}] n={len(y)}", flush=True)
        m = _metrics(X, y, args.seed, args.njobs)
        summary[name] = m
        print("      t-SNE ...", end="", flush=True)
        E = _tsne(X, args.seed)
        print(" done", flush=True)
        for t in sorted(set(y)):
            s = y == t
            ax.scatter(E[s, 0], E[s, 1], s=16, alpha=0.75,
                       color=colour[t], edgecolors="none", label=PRETTY.get(t, t))
        # No figure-level title: the caption carries the claim, and a suptitle
        # only pushes the panels down and duplicates it at an unreadable size.
        # The metrics go under the panel instead of onto a second title line,
        # which at a legible size runs wider than the panel and collides with
        # the neighbouring one.
        ax.set_title(name, fontsize=19, pad=10)
        ax.set_xlabel(
            f"silhouette={m['silhouette_128d']:+.3f}    "
            f"kNN={m['knn_accuracy_mean']:.3f}\u00b1{m['knn_accuracy_std']:.3f}    "
            f"probe={m['linear_probe_accuracy_mean']:.3f}",
            fontsize=14, labelpad=8)
        ax.set_xticks([]); ax.set_yticks([])

    # Legend below both panels: in-axes it sits on top of the embedding, and
    # there is no empty corner that stays empty for every seed.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=15, markerscale=2.4, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.085, 1, 1])
    p = os.path.join(
        args.out,
        f"prefusion_vs_adapter_{args.hand_name}_"
        f"{FINGER_NAMES[args.finger]}.{args.ext}")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[plot] wrote {p}")

    js = os.path.join(args.out, "summary_prefusion.json")
    with open(js, "w") as f:
        json.dump({"hand_slot": args.hand_slot, "hand_name": args.hand_name,
                   "finger": FINGER_NAMES[args.finger], "panels": summary},
                  f, indent=2)
    print(f"[plot] wrote {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
