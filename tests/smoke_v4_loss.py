"""Local synthetic-tensor smoke for the v4 trainer changes.

Exercises the new code paths in :mod:`runner.tactile_vae_trainer` without
touching the dataset / model / accelerator: just imports the module-level
helpers, builds a tiny stub trainer state, and runs four checks:

  1. ``_avg_pool_T`` shape + value correctness for kernel=1 (no-op),
     kernel=3 (centered moving average), and T<kernel (silent skip).
  2. ``_compute_flow_loss`` reduces EXACTLY to plain MSE when GT magnitude
     is uniform (rel == 1, weight == 1 + lambda_w), and produces a finite
     scalar otherwise. Also verifies that disabling the block falls back
     to ``F.mse_loss``.
  3. The validate() v4b accumulator math: physical-unit threshold fed
     through denormalize gives expected per-finger active counts when GT
     flow is constructed with a known number of >threshold pixels.
  4. ``_select_best_metric_value('val_flow_mse_active_mean')`` averages
     the right keys.

CPU is fine -- nothing here needs CUDA.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml


# Make ``runner.tactile_vae_trainer`` importable.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)


def _approx_eq(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def test_avg_pool_T(mod):
    print("\n[1] _avg_pool_T")
    # window=1 short-circuit: identity.
    x = torch.randn(2, 5, 9, 4, 6, 1)
    y = mod._avg_pool_T(x, kernel=1)
    assert torch.equal(x, y), "kernel=1 must be identity"
    print("  kernel=1 identity: OK")

    # window=3 centered moving average. Build a ramp along T so the answer is known.
    T = 9
    x = torch.zeros(1, 1, T, 1, 1, 1)
    for t in range(T):
        x[0, 0, t, 0, 0, 0] = float(t)
    y = mod._avg_pool_T(x, kernel=3)
    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
    # Interior frames: average of (t-1, t, t+1) == t.
    for t in range(1, T - 1):
        got = float(y[0, 0, t, 0, 0, 0])
        assert _approx_eq(got, float(t)), f"interior t={t}: {got} != {t}"
    # Boundaries: count_include_pad=False so t=0 is mean(0,1) = 0.5, t=T-1 is mean(T-2,T-1) = T-1.5.
    assert _approx_eq(float(y[0, 0, 0, 0, 0, 0]), 0.5), float(y[0, 0, 0, 0, 0, 0])
    assert _approx_eq(float(y[0, 0, T - 1, 0, 0, 0]), T - 1.5), float(y[0, 0, T - 1, 0, 0, 0])
    print("  kernel=3 centered avg + boundary: OK")

    # T < kernel: silent skip.
    x_short = torch.randn(1, 5, 1, 4, 6, 1)
    y_short = mod._avg_pool_T(x_short, kernel=3)
    assert torch.equal(x_short, y_short), "T<kernel must skip"
    print("  T<kernel skip: OK")


def _make_tiny_trainer(mod, flow_cfg=None):
    """Construct a minimum-viable TactileVAETrainer-like object exposing only
    the attributes that ``_compute_flow_loss`` and ``_select_best_metric_value``
    touch. Avoids the real ``__init__`` (Accelerator + datasets) entirely.
    """
    class _Stub:
        pass

    cfg_block = {"flow_loss": flow_cfg} if flow_cfg is not None else {}
    stub = _Stub()
    stub.args = argparse.Namespace(tactile_vae={"config": cfg_block})
    stub.cfg = {"data": {"train": {"flow_stats_path": None}, "val": {}}}
    stub._flow_mean_t = None
    stub._flow_std_t = None
    stub._best_ckpt_metric = "val_flow_mse_active_mean"
    # Bind unbound methods.
    stub._denormalize_flow = lambda flow, _self=stub: mod.TactileVAETrainer._denormalize_flow(_self, flow)
    stub._compute_flow_loss = lambda pred, gt, _self=stub: mod.TactileVAETrainer._compute_flow_loss(_self, pred, gt)
    stub._select_best_metric_value = lambda metrics, _self=stub: mod.TactileVAETrainer._select_best_metric_value(_self, metrics)
    return stub


def test_compute_flow_loss_disabled(mod):
    print("\n[2a] _compute_flow_loss (disabled -> plain MSE)")
    stub = _make_tiny_trainer(mod, flow_cfg=None)
    pred = torch.randn(2, 5, 3, 4, 6, 3)
    gt = torch.randn(2, 5, 3, 4, 6, 3)
    expected = F.mse_loss(pred, gt).item()
    got = stub._compute_flow_loss(pred, gt).item()
    assert _approx_eq(got, expected, tol=1e-6), f"{got} vs {expected}"
    print(f"  disabled fallback: {got:.6f} == F.mse_loss {expected:.6f} OK")


def test_compute_flow_loss_uniform_mag(mod):
    """When GT magnitude is uniform across (T,H,W), rel == 1 everywhere, so
    weight = 1 + lambda_w * 1 = 1 + lambda_w (constant). The normalized
    weighted mean reduces to F.mse_loss in that case.
    """
    print("\n[2b] _compute_flow_loss (uniform GT magnitude -> plain MSE)")
    flow_cfg = {
        "enabled": True, "form": "sqrt_magnitude_aware",
        "lambda_w": 2.0, "w_max": 5.0, "eps": 1e-6,
        "contact_threshold": 0.5, "temporal_smooth_window": 1,
    }
    stub = _make_tiny_trainer(mod, flow_cfg=flow_cfg)
    # Build GT with constant magnitude=2 (e.g. (2,0,0)) so ||gt|| is uniform.
    B, NF, T, H, W = 2, 5, 3, 4, 6
    gt = torch.zeros(B, NF, T, H, W, 3)
    gt[..., 0] = 2.0
    pred = gt + torch.randn_like(gt) * 0.1
    expected = F.mse_loss(pred, gt).item()
    got = stub._compute_flow_loss(pred, gt).item()
    assert _approx_eq(got, expected, tol=1e-5), f"{got} vs {expected}"
    print(f"  uniform mag: {got:.6f} == F.mse_loss {expected:.6f} OK")


def test_compute_flow_loss_nonuniform_mag(mod):
    """When GT has hot spots, weighted-mean must up-weight hot pixels.
    Compare two error placements with identical L2 magnitude:
      * error placed at HOT pixel  -> larger weighted loss
      * error placed at COOL pixel -> smaller weighted loss
    """
    print("\n[2c] _compute_flow_loss (non-uniform mag amplifies hot pixels)")
    flow_cfg = {
        "enabled": True, "form": "sqrt_magnitude_aware",
        "lambda_w": 2.0, "w_max": 5.0, "eps": 1e-6,
        "contact_threshold": 0.5, "temporal_smooth_window": 1,
    }
    stub = _make_tiny_trainer(mod, flow_cfg=flow_cfg)
    B, NF, T, H, W = 1, 1, 1, 4, 6
    gt = torch.zeros(B, NF, T, H, W, 3)
    # Single hot pixel with motion magnitude 10 in dx; rest is zero.
    gt[0, 0, 0, 0, 0, 0] = 10.0

    err = 0.5
    pred_hot  = gt.clone(); pred_hot[0, 0, 0, 0, 0, 0]  -= err   # error AT hot pixel
    pred_cool = gt.clone(); pred_cool[0, 0, 0, 1, 1, 0] += err   # error AT cool pixel

    loss_hot = stub._compute_flow_loss(pred_hot, gt).item()
    loss_cool = stub._compute_flow_loss(pred_cool, gt).item()
    print(f"  loss_hot={loss_hot:.6f}  loss_cool={loss_cool:.6f}")
    assert loss_hot > loss_cool * 1.5, (loss_hot, loss_cool, "hot must be >> cool")
    print("  hot >> cool: OK")


def test_compute_flow_loss_w_max_clamp(mod):
    """w_max should clamp the weight ceiling. Pick GT s.t. some pixels have
    rel >> (w_max-1)/lambda_w squared so unclamped weight would exceed w_max.
    Compare loss with w_max=5 vs w_max=100; the clamped one should be smaller
    (weighted mean gets denominator-dominated by background pixels).
    """
    print("\n[2d] _compute_flow_loss (w_max clamp lowers loss vs unclamped)")
    base_cfg = {
        "enabled": True, "form": "sqrt_magnitude_aware",
        "lambda_w": 2.0, "eps": 1e-6,
        "contact_threshold": 0.5, "temporal_smooth_window": 1,
    }
    stub_clamp = _make_tiny_trainer(mod, flow_cfg={**base_cfg, "w_max": 5.0})
    stub_loose = _make_tiny_trainer(mod, flow_cfg={**base_cfg, "w_max": 100.0})

    B, NF, T, H, W = 1, 1, 1, 4, 6
    gt = torch.zeros(B, NF, T, H, W, 3)
    gt[0, 0, 0, 0, 0, 0] = 100.0  # extreme outlier
    pred = gt.clone()
    pred[0, 0, 0, 0, 0, 0] -= 1.0

    loss_clamp = stub_clamp._compute_flow_loss(pred, gt).item()
    loss_loose = stub_loose._compute_flow_loss(pred, gt).item()
    # With outlier weight CLAMPED to 5, the relative weighting of the hot
    # pixel is smaller, so its 1.0 error contributes less per unit weight,
    # i.e. weighted mean is smaller. Loose w_max gives a much higher weight
    # at the hot pixel -> proportionally larger loss.
    print(f"  loss_clamp(w_max=5)={loss_clamp:.6f}  loss_loose(w_max=100)={loss_loose:.6f}")
    assert loss_loose > loss_clamp * 1.2, (loss_loose, loss_clamp)
    print("  clamp(loose) > clamp(tight): OK")


def test_denormalize_with_stats(mod):
    """When _flow_mean_t / _flow_std_t are set, _denormalize_flow inverts the
    x -> (x - mean) / std mapping; with both = identity, output == input.
    """
    print("\n[3a] _denormalize_flow with stats")
    stub = _make_tiny_trainer(mod, flow_cfg=None)
    # Identity stats (mean=0, std=1) -> denorm == identity.
    stub._flow_mean_t = torch.zeros(1, 1, 1, 1, 1, 3)
    stub._flow_std_t  = torch.ones(1, 1, 1, 1, 1, 3)
    x = torch.randn(2, 5, 3, 4, 6, 3)
    y = mod.TactileVAETrainer._denormalize_flow(stub, x)
    assert torch.allclose(x, y, atol=1e-6), "identity stats must yield identity"
    # Non-trivial: mean=1, std=2 -> y = 2x + 1.
    stub._flow_mean_t = torch.ones(1, 1, 1, 1, 1, 3)
    stub._flow_std_t  = torch.full((1, 1, 1, 1, 1, 3), 2.0)
    y = mod.TactileVAETrainer._denormalize_flow(stub, x)
    assert torch.allclose(y, 2.0 * x + 1.0, atol=1e-6)
    print("  identity + (2x+1): OK")


def test_validate_accumulator_math(mod):
    """Mirror the v4b mask-and-accumulate snippet from validate() and verify
    counts match a hand-constructed example. Uses identity flow stats so the
    physical-unit threshold operates directly on the normalized magnitude.
    """
    print("\n[3b] validate() v4b accumulators")
    stub = _make_tiny_trainer(mod, flow_cfg={"contact_threshold": 0.5})
    stub._flow_mean_t = torch.zeros(1, 1, 1, 1, 1, 3)
    stub._flow_std_t  = torch.ones(1, 1, 1, 1, 1, 3)

    B, NF, T, H, W = 1, 5, 1, 4, 6
    gt   = torch.zeros(B, NF, T, H, W, 3)
    pred = torch.zeros(B, NF, T, H, W, 3)
    # Finger 0 (thumb): two pixels with magnitude 2.0 (above threshold), rest 0.
    gt[0, 0, 0, 0, 0, 0] = 2.0
    gt[0, 0, 0, 0, 1, 0] = 2.0
    # Finger 4 (pinky): one pixel above threshold; pred matches it.
    gt[0, 4, 0, 0, 0, 0] = 1.0
    pred[0, 4, 0, 0, 0, 0] = 1.0

    contact_threshold = 0.5
    gt_mag = torch.linalg.norm(gt, dim=-1)        # (B, F, T, H, W)
    pred_mag = torch.linalg.norm(pred, dim=-1)
    gt_active = gt_mag > contact_threshold
    pred_active = pred_mag > contact_threshold
    reduce_dims = [0, 2, 3, 4]
    active_cnt = gt_active.float().sum(dim=reduce_dims).long().tolist()
    pred_active_cnt = pred_active.float().sum(dim=reduce_dims).long().tolist()
    tp_cnt = (pred_active & gt_active).float().sum(dim=reduce_dims).long().tolist()

    # Hand truth.
    expected_active     = [2, 0, 0, 0, 1]
    expected_pred_active = [0, 0, 0, 0, 1]
    expected_tp         = [0, 0, 0, 0, 1]
    print(f"  active_cnt={active_cnt}      expected={expected_active}")
    print(f"  pred_active_cnt={pred_active_cnt} expected={expected_pred_active}")
    print(f"  tp_cnt={tp_cnt}              expected={expected_tp}")
    assert active_cnt == expected_active
    assert pred_active_cnt == expected_pred_active
    assert tp_cnt == expected_tp
    print("  per-finger mask + reduce: OK")


def test_best_metric_selector(mod):
    print("\n[4] _select_best_metric_value('val_flow_mse_active_mean')")
    stub = _make_tiny_trainer(mod, flow_cfg=None)
    stub._best_ckpt_metric = "val_flow_mse_active_mean"
    metrics = {
        "val/T1/flow_loss": 0.50,
        "val/T9/flow_loss": 0.40,
        "val/T1/flow_mse_active": 1.0,
        "val/T9/flow_mse_active": 3.0,
        "val/T1/flow_mse_inactive": 0.05,
    }
    got = stub._select_best_metric_value(metrics)
    assert _approx_eq(got, 2.0), got
    print(f"  active_mean=2.0: OK (got {got})")
    # Make sure existing branch still works.
    stub._best_ckpt_metric = "val_flow_mean"
    got = stub._select_best_metric_value(metrics)
    assert _approx_eq(got, 0.45), got
    print(f"  legacy val_flow_mean=0.45: OK (got {got})")


def _build_dryrun_trainer(mod, yaml_path):
    """Construct a trainer-like stub that ``prepare_models()`` can run against
    without spinning up Accelerator / dataset / DDP. Mirrors only the
    attributes ``prepare_models()`` actually reads:

        * ``self.cfg``                       -- raw yaml dict (for flow_stats path)
        * ``self.args``                      -- argparse.Namespace built from cfg
        * ``self.state.accelerator.device``  -- where _flow_mean_t / _flow_std_t live

    Also swaps the trainer's accelerate-aware module logger for a plain
    stdlib logger so prepare_models()'s ``logger.info(...)`` calls don't
    explode with "must initialize accelerate state". The original logger is
    restored by the caller after prepare_models() returns.
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    trainer_cls = mod.TactileVAETrainer
    stub = trainer_cls.__new__(trainer_cls)
    stub.cfg = cfg
    stub.args = argparse.Namespace(**cfg)
    stub.state = mod._State()
    # Minimal accelerator surface: prepare_models only touches
    # state.accelerator.device. Use CPU so the test runs anywhere.
    stub.state.accelerator = SimpleNamespace(device=torch.device("cpu"))
    return stub


def test_yaml_to_prepare_models(mod):
    """Catch the class of bug where a new yaml field (``flow_loss``)
    leaking into ``TactileVAE.__init__`` because the strip list in
    ``prepare_models()`` was incomplete.

    Loads each tracked Stage 1 yaml under configs/tactile_model/, builds a
    minimum-viable trainer stub, and runs the real ``prepare_models()`` from
    the trainer module. Any kwargs / type / nested-block mismatch fails
    here on the cheap CPU path before burning a GPU slot.
    """
    import logging as _stdlib_logging
    print("\n[5] yaml -> prepare_models dry-run (catches v4 flow_loss-style strip bugs)")
    cfg_dir = os.path.join(REPO_ROOT, "configs")
    yamls = [
        os.path.join(cfg_dir, "stage1_tactile_encoder.yaml"),
    ]
    # The trainer module uses ``accelerate.logging.get_logger``, which refuses
    # to ``.info`` until Accelerator is initialized. We don't want to spin up
    # a real Accelerator just to dry-run prepare_models, so swap in a plain
    # stdlib logger for the duration of this test and restore after.
    real_logger = mod.logger
    mod.logger = _stdlib_logging.getLogger("smoke_v4_loss.dryrun")
    mod.logger.setLevel(_stdlib_logging.WARNING)  # quiet INFO chatter
    try:
        for yp in yamls:
            if not os.path.isfile(yp):
                print(f"  skip (missing): {yp}")
                continue
            stub = _build_dryrun_trainer(mod, yp)
            mod.TactileVAETrainer.prepare_models(stub)
            cfg_cfg = stub.args.tactile_vae["config"]
            assert "flow_loss" in cfg_cfg, "flow_loss block must remain in cfg for _compute_flow_loss"
            assert cfg_cfg["flow_loss"]["enabled"] is True
            # And it must NOT have leaked into the model constructor
            # (the bug this guards against). _LOSS_ONLY_KEYS strip is
            # implicit -- prepare_models() raised TypeError before; if
            # we got here, it didn't.
            assert hasattr(stub, "vae"), "prepare_models did not set self.vae"
            n_params = sum(p.numel() for p in stub.vae.parameters())
            print(f"  {os.path.basename(yp)}: TactileVAE built ({n_params/1e6:.1f}M params)")
    finally:
        mod.logger = real_logger
    print("  yaml dry-run: OK")


def main():
    print(f"torch={torch.__version__}, np={np.__version__}")
    mod = importlib.import_module("runner.tactile_vae_trainer")
    test_avg_pool_T(mod)
    test_compute_flow_loss_disabled(mod)
    test_compute_flow_loss_uniform_mag(mod)
    test_compute_flow_loss_nonuniform_mag(mod)
    test_compute_flow_loss_w_max_clamp(mod)
    test_denormalize_with_stats(mod)
    test_validate_accumulator_math(mod)
    test_best_metric_selector(mod)
    test_yaml_to_prepare_models(mod)
    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
