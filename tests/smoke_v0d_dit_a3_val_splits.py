"""Phase A3 smoke: multi-split val_loaders end-to-end.

What this proves about Phase A3 (multi-split validation dataloaders):

  1. ``runner.val_loaders`` is materialized from yaml ``data.val_splits``
     with the named splits the yaml configured. The smoke is yaml-
     agnostic: ``expected_names`` is read from the yaml itself, so this
     same script works for the cube-only dryrun (`{"cube_val"}`) AND
     the production 488-midtraining yaml which will add `holdout_488`
     (`{"cube_val", "holdout_488"}`). See A3.4 note: the cube dryrun
     intentionally ships with cube_val only to keep the multi-split
     plumbing smoke decoupled from the 488 corpus bring-up.
  2. Each split's underlying dataset is non-empty AND
     ``set(dataset._kept_episode_indices)`` is EXACTLY the
     ``episodes`` list configured in yaml -- catches global-vs-local
     episode_index confusion (a real failure mode when ids look like
     [90..99] but the corpus uses absolute indexing).
  3. ``runner._compute_val_loss(loader, tag=split_name, max_batches=1)``
     runs end-to-end on EVERY split with finite, non-negative losses
     for all 5 components (loss / loss_video / loss_visual /
     loss_tactile / loss_action) and the correct ``num_batches``.
  4. After back-to-back calls across ALL splits, the 4-stream RNG
     (torch CPU + torch CUDA + Python ``random`` + ``numpy.random``)
     is BIT-EQUAL to the pre-loop snapshot. Extends the A2 per-call
     RNG check to "loops do not pollute even across multiple calls",
     which is exactly the scenario the production train loop hits at
     every ``steps_to_val`` interval.
  5. When the yaml has ``adapter_use_pose_injection=True`` (v0d
     mode), ``runner._v0d_gate1_encode_logged`` is True after the
     first split's val call. This proves the val codepath went
     through ``_encode_tactile_split`` with hand_pose, not the v0c-A
     fallback -- a silent regression here would mean train and val
     use different forward passes and the gen-gap study would be
     uninterpretable.

Run as:

    python scripts/smoke_v0d_dit_a3_val_splits.py \
        --config configs/cube_handover/stage2_world_model.yaml \
        [--seed 42]

Exit code 0 = PASS, 1 = FAIL or ERROR.

Known limitation (deliberately out of A3 scope): the val DataLoader
has no ``DistributedSampler``. In multi-rank runs, every rank
iterates the full val set. ``_compute_val_loss`` uses
``accelerator.reduce(..., reduction='mean')`` so the AGGREGATE loss
is correct; the redundant compute only costs val wall-clock and is
fine until val time becomes a bottleneck (currently dominated by
train-step time). Optimize with a distributed val sampler in a
later phase if needed.
"""

import argparse
import os
import sys


# Reuse helpers from the A2 numerical smoke. Direct import keeps the
# wiring simple; only worth extracting into a shared `scripts/_smoke_utils.py`
# module if a third smoke script materializes.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from smoke_v0d_dit_a2_numerical import (  # noqa: E402
    _ensure_project_root_on_path,
    _override_config_for_smoke,
    _force_global_determinism,
    _ensure_dist_initialized,
)


def _run_a3_smoke(config_path: str, seed: int) -> int:
    """Build trainer with val_splits yaml, exercise each loader, verify."""
    _ensure_project_root_on_path()
    _force_global_determinism(seed)
    _ensure_dist_initialized()

    # Smoke yaml overrides (single-GPU, no DeepSpeed, no augmentations).
    # train_steps_cap doesn't matter here -- we never call runner.train(),
    # only build the runner and exercise val_loaders directly.
    smoke_yaml = _override_config_for_smoke(
        config_path,
        seed=seed,
        batch_size=2,
        train_steps_cap=1,
    )

    import torch
    from utils import import_custom_class

    # Re-read smoke yaml to learn what splits + episode lists the user
    # asked for; the smoke is yaml-agnostic so it works with whatever
    # set of split names the config defines.
    import yaml
    with open(smoke_yaml) as f:
        cfg = yaml.safe_load(f)
    expected_splits = cfg.get("data", {}).get("val_splits", {}) or {}
    if not expected_splits:
        print(
            f"[a3-smoke FAIL] config {config_path!r} has no "
            f"data.val_splits block. Nothing to test."
        )
        return 1
    expected_names = set(expected_splits.keys())
    print(f"[a3-smoke] expected val_splits: {sorted(expected_names)}")

    Runner = import_custom_class(
        "TactileDiTTrainer", "runner/tactile_dit_trainer.py",
    )
    runner = Runner(smoke_yaml)
    runner.prepare_dataset()
    runner.prepare_models()
    runner.prepare_trainable_parameters()
    runner.prepare_optimizer()
    runner.prepare_for_training()
    runner.prepare_trackers()

    # Same projector dtype fixup as the A2 smoke -- single-GPU mode
    # bypasses DeepSpeed's automatic cast, so the projector stays in
    # fp32 and would crash with `expected BFloat16 but found Float`
    # on the first forward.
    weight_dtype = runner.state.weight_dtype
    if runner.projector is not None and weight_dtype != torch.float32:
        runner.projector.to(dtype=weight_dtype)
    if (
        getattr(runner, "tactile_vae", None) is not None
        and weight_dtype != torch.float32
    ):
        runner.tactile_vae.to(dtype=weight_dtype)

    # ---- Assertion 1: val_loaders has the expected named splits ----
    actual_names = set(runner.val_loaders.keys())
    assert actual_names == expected_names, (
        f"runner.val_loaders keys {sorted(actual_names)} != "
        f"expected {sorted(expected_names)}; prepare_val_splits did "
        f"not materialize the configured splits."
    )
    print(
        f"[a3-smoke] assertion 1 PASS: val_loaders keys == "
        f"{sorted(actual_names)}"
    )

    # ---- Assertion 2: per-split episode set matches yaml exactly ----
    for name, loader in runner.val_loaders.items():
        ds = loader.dataset
        n = len(ds)
        assert n > 0, (
            f"val_loaders[{name!r}].dataset has len 0; "
            f"prepare_val_splits returned an empty dataset."
        )
        configured = {int(e) for e in expected_splits[name].get("episodes", [])}
        kept = getattr(ds, "_kept_episode_indices", None)
        assert kept is not None, (
            f"val_loaders[{name!r}].dataset._kept_episode_indices is "
            f"None; A3.0 dataset-filter wiring failed for this split."
        )
        assert kept == configured, (
            f"val_loaders[{name!r}] kept episodes != configured.\n"
            f"  configured (n={len(configured)}): "
            f"{sorted(configured)[:6]}"
            f"{'...' if len(configured) > 6 else ''}\n"
            f"  kept       (n={len(kept)}): "
            f"{sorted(kept)[:6]}"
            f"{'...' if len(kept) > 6 else ''}\n"
            f"  in_configured_not_kept: {sorted(configured - kept)}\n"
            f"  in_kept_not_configured: {sorted(kept - configured)}\n"
            f"Likely cause: global-vs-local episode_index confusion "
            f"or wrong data_roots."
        )
        print(
            f"[a3-smoke] assertion 2 PASS for {name!r}: "
            f"len(dataset)={n}, kept_episodes={len(kept)} "
            f"(configured={len(configured)})"
        )

    # ---- Pre-call 4-stream RNG snapshot (for Assertion 4) ----
    # Re-seed first so the snapshot is from a known baseline (the
    # prepare_* chain consumes RNG: random.randint inside
    # prepare_val_dataset, dataloader worker seeds, model
    # from_pretrained, accelerator prep, ...).
    import random as _py_random
    import numpy as _np
    _force_global_determinism(seed)
    rng_cpu_before = torch.get_rng_state()
    rng_cuda_before = (
        [torch.cuda.get_rng_state(d)
         for d in range(torch.cuda.device_count())]
        if torch.cuda.is_available() else None
    )
    rng_py_before = _py_random.getstate()
    rng_np_before = _np.random.get_state()

    # ---- Assertion 3: each split runs end-to-end with finite loss ----
    # Lock starting train() mode the same way the real train loop does
    # (per epoch), so _compute_val_loss has something to restore to.
    runner._prepared_model.train()
    assert runner._prepared_model.training is True
    results = {}
    for name, loader in runner.val_loaders.items():
        print(
            f"[a3-smoke] invoking _compute_val_loss(tag={name!r}, "
            f"max_batches=1) ..."
        )
        # CRITICAL: tag=split_name (NO 'val/' prefix). The helper
        # formats `[val/{tag}] ...` internally; passing tag='val/foo'
        # would double-prefix to `[val/val/foo]`. See A3 plan
        # post-review adjustment #2.
        result = runner._compute_val_loss(loader, max_batches=1, tag=name)
        results[name] = result
        for k in ("loss", "loss_video", "loss_visual",
                  "loss_tactile", "loss_action"):
            v = result[k]
            assert isinstance(v, float), (
                f"results[{name!r}][{k!r}] = {v!r}; expected float."
            )
            # NaN fails `v >= 0.0` so this also catches NaN losses.
            assert v >= 0.0, (
                f"results[{name!r}][{k!r}] = {v} is negative or NaN."
            )
        assert result["num_batches"] == 1, (
            f"results[{name!r}]['num_batches'] = {result['num_batches']}; "
            f"expected 1 (max_batches=1)."
        )
        print(
            f"[a3-smoke] assertion 3 PASS for {name!r}: "
            f"loss={result['loss']:.6f} "
            f"num_batches={result['num_batches']}"
        )

    # ---- Assertion 4: all 4 RNG streams restored after LOOP ----
    # This is stricter than A2's per-call check: we assert the FULL
    # loop over every split leaves the global RNGs identical to the
    # pre-loop snapshot. The production train loop hits this exact
    # pattern at every steps_to_val interval.
    assert torch.equal(rng_cpu_before, torch.get_rng_state()), (
        "CPU RNG state changed after multi-split val loop. "
        "RNG leak: train RNG would drift after each val interval."
    )
    if rng_cuda_before is not None:
        for d, state_before in enumerate(rng_cuda_before):
            assert torch.equal(state_before, torch.cuda.get_rng_state(d)), (
                f"CUDA RNG device {d} state changed after multi-split val loop."
            )
    assert _py_random.getstate() == rng_py_before, (
        "Python `random` state changed after multi-split val loop "
        "(dataloader.get_frame_indexes uses random.randint, so a "
        "drift here corrupts subsequent train-step batch indices)."
    )
    np_state_after = _np.random.get_state()
    assert np_state_after[0] == rng_np_before[0]
    assert _np.array_equal(np_state_after[1], rng_np_before[1]), (
        "numpy `np.random` state changed after multi-split val loop "
        "(dataloader.get_frame_indexes uses np.random.choice, so a "
        "drift here corrupts subsequent train-step memory indices)."
    )
    assert np_state_after[2:] == rng_np_before[2:]
    print(
        "[a3-smoke] assertion 4 PASS: 4-stream RNG restored after "
        "looping over all val_splits."
    )

    # ---- Assertion 5: v0d Gate-1 encode flag (conditional) ----
    use_pose_injection = bool(
        cfg.get("tactile_vae", {})
           .get("config", {})
           .get("adapter_use_pose_injection", False)
    )
    if use_pose_injection:
        assert getattr(runner, "_v0d_gate1_encode_logged", None) is True, (
            "v0d mode (adapter_use_pose_injection=True in yaml) but "
            "_v0d_gate1_encode_logged is False after multi-split val. "
            "The val codepath did NOT enter _encode_tactile_split with "
            "v0d kwargs; it silently fell back to v0c-A. 488 midtraining "
            "val curves would be on a different forward pass than train, "
            "invalidating the gen-gap comparison."
        )
        print(
            "[a3-smoke] assertion 5 PASS: v0d Gate-1 encode flag is True "
            "(val codepath went through v0d encode with hand_pose)."
        )
    else:
        print(
            "[a3-smoke] assertion 5 SKIPPED: yaml has "
            "adapter_use_pose_injection=False (v0c-A mode)."
        )

    # ---- Final PASS banner ----
    print()
    print("[a3-smoke PASS] multi-split val_loaders verified end-to-end:")
    print(f"  - val_loaders keys = {sorted(actual_names)}")
    for name in sorted(actual_names):
        ds = runner.val_loaders[name].dataset
        n_kept = len(ds._kept_episode_indices)
        n_samples = len(ds)
        loss = results[name]["loss"]
        print(
            f"  - {name:<14} : {n_samples:>5} samples "
            f"({n_kept:>3} kept episodes), loss[1 batch] = {loss:.6f}"
        )
    print(
        "  - 4-stream RNG (torch CPU/CUDA + Python random + numpy) "
        "restored after all calls"
    )
    if use_pose_injection:
        print(
            "  - [v0d] _v0d_gate1_encode_logged confirmed True "
            "from val codepath"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase A3 smoke for TactileDiTTrainer.val_loaders.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to yaml with data.val_splits block (typically the "
             "v0d cube dryrun).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global RNG seed for the smoke (default: 42).",
    )
    args = parser.parse_args()
    try:
        return _run_a3_smoke(args.config, args.seed)
    except AssertionError as e:
        print(f"\n[a3-smoke FAIL] {e}", file=sys.stderr)
        return 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(
            f"\n[a3-smoke ERROR] unexpected "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
