"""Stage 2 F1: structural + invariant smoke for ``TactileProjector``.

Mirrors the structure of ``scripts/smoke_visual_vae_adapter.py``: each check is
an isolated, deterministic function that asserts a single contract; the driver
runs them in order and prints a short banner per check. CPU-runnable; the
autocast check is skipped automatically if CUDA is unavailable.

Checks (each contract from Section 3 of the locked Stage-2 spec):

  1. Shape contract: ``(2, 2, 128, 5, 6, 8)`` in -> same shape out (defaults).
  2. Init invariant -- alpha=0 means at step 0::

         out = x + modality_bias + view_bias[view_idx]

     i.e. the residual MLP path contributes EXACTLY zero (because
     ``alpha * mlp_out = 0 * non-zero = 0`` exactly, regardless of mlp_out).
  3. view_idx routes: feeding ``[[0, 1]]`` vs ``[[1, 0]]`` produces different
     outputs (proves view_embed actually disambiguates hands).
  4. Gradient flow (two-phase, mirrors FingerSetTransformerAdapter):

       4A. At canonical init (alpha=0), the AdaLN-Zero invariant holds:
           ONLY direct-path params (``alpha``, ``modality_bias.bias``,
           ``view_embed.weight``) get non-zero gradient. ``norm.*`` and
           ``mlp.*`` are blocked by the ``alpha=0`` factor at the residual
           junction -- this is the intended warm-start behavior.

       4B. After bumping alpha to 0.1 + redo forward/backward, ALL params
           (``norm.*``, ``mlp.*``, ``alpha``, ``modality_bias.bias``,
           ``view_embed.weight``) get non-zero gradient. Confirms the full
           graph is wired and only the gate is what gates it.
  5. Autocast bf16: under ``torch.autocast(bf16)``, forward succeeds (no
     crash) and backward populates fp32 master grads. NOTE: the output
     dtype may be fp32 because LayerNorm is autocast-fp32 and the residual
     ``x + alpha * mlp_out`` adds an fp32 tensor to a bf16 tensor (the
     result auto-promotes to fp32). This is the standard mixed-precision
     pattern; what matters for memory savings is that the matmuls inside
     the MLP run in bf16, which we verify by passing bf16 input and
     confirming an intermediate bf16 path. SKIPPED if CUDA unavailable.
  6. Param count: total trainable params < 100K (Section 3 budget ~66.5K).
  7. ``num_views=10`` forward-compat: ``view_embed.weight.shape == (10, 128)``
     and forward at ``V_hand=10`` round-trips the shape correctly. This
     guards Stage-3 per-finger view ablations against silent regressions.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from models.tactile_models.projector import TactileProjector  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _section(title: str) -> None:
    print(f"\n[{title}]")


def _make_default_projector(seed: int = 0) -> TactileProjector:
    torch.manual_seed(seed)
    return TactileProjector(latent_dim=128, hidden_dim=256, num_views=2)


def _ascending_view_idx(b: int, v: int) -> torch.Tensor:
    """``[[0, 1, ..., v-1], ...] * b`` -- the canonical (B, V_hand) layout."""
    return torch.arange(v, dtype=torch.int64).unsqueeze(0).expand(b, v).contiguous()


# ---------------------------------------------------------------------------
# 1. Shape contract
# ---------------------------------------------------------------------------


def test_shape_contract():
    _section("1] Shape contract: (2, 2, 128, 5, 6, 8) in -> out same shape")
    proj = _make_default_projector()
    proj.eval()
    x = torch.randn(2, 2, 128, 5, 6, 8)
    view_idx = _ascending_view_idx(2, 2)
    with torch.no_grad():
        out = proj(x, view_idx)
    assert out.shape == x.shape, (
        f"output shape {tuple(out.shape)} != input shape {tuple(x.shape)}"
    )
    print(f"  in/out shape = {tuple(out.shape)} OK")


# ---------------------------------------------------------------------------
# 2. Init invariant: alpha=0 -> out == x + modality_bias + view_bias[view_idx]
# ---------------------------------------------------------------------------


def test_init_invariant_residual_is_zero():
    _section("2] alpha=0 -> out == x + modality + view_bias EXACTLY")
    proj = _make_default_projector()
    proj.eval()

    # Sanity: alpha is exactly zero at init. This is the AdaLN-Zero gate.
    assert float(proj.alpha) == 0.0, f"alpha should be 0; got {float(proj.alpha)}"

    # Sanity: the last MLP linear is INTENTIONALLY NOT zero-init (Kaiming).
    # If both alpha=0 and mlp[-1] were zero-init, alpha grad = upstream * 0 = 0
    # at step 0 and the projector would be stuck at the identity bias forever.
    # With Kaiming init for mlp[-1], mlp_out at init is non-zero -> alpha gets
    # immediate non-zero gradient -> alpha leaves zero -> the rest of MLP /
    # LayerNorm starts receiving gradient. See projector.py for the full note.
    assert proj.mlp[-1].weight.abs().max().item() > 0, (
        "mlp[-1].weight is zero-init; combined with alpha=0 this is a dead "
        "state. Switch to Kaiming-init for mlp[-1] (the default)."
    )

    x = torch.randn(2, 2, 128, 5, 6, 8)
    view_idx = _ascending_view_idx(2, 2)
    with torch.no_grad():
        out = proj(x, view_idx)
        # Even though mlp_out != 0, alpha=0 makes alpha*mlp_out exactly zero.
        # Therefore: out = x + modality_bias + view_embed[view_idx].
        m_bias = proj.modality_bias.bias.view(1, 1, -1, 1, 1, 1)
        v_bias = proj.view_embed(view_idx).view(2, 2, 128, 1, 1, 1)
        expected = x + m_bias + v_bias

    diff = (out - expected).abs().max().item()
    print(f"  max|out - (x + modality + view_bias)| = {diff:.3e}")
    assert diff < 1e-6, (
        f"residual MLP path is not exactly zero at init (diff={diff:.3e}). "
        f"alpha-zero gate is broken."
    )

    # Bump alpha and confirm output diverges -- proves residual is wired.
    with torch.no_grad():
        proj.alpha.fill_(1.0)
        out_nonzero = proj(x, view_idx)
    delta = (out_nonzero - expected).abs().max().item()
    print(f"  alpha=1: max|out - expected_init| = {delta:.3e}")
    assert delta > 1e-3, (
        f"alpha=1 should change output noticeably; got {delta:.3e}. "
        f"Residual path may not be wired."
    )


# ---------------------------------------------------------------------------
# 3. view_idx routes hand differentiation
# ---------------------------------------------------------------------------


def test_view_idx_routes_distinguish_hands():
    _section("3] view_idx routes: [0, 1] vs [1, 0] -> different outputs")
    proj = _make_default_projector(seed=42)
    proj.eval()
    # Make view_embed entries clearly different (Gaussian init is already
    # non-zero, but let's be paranoid -- give them a wide spread).
    with torch.no_grad():
        proj.view_embed.weight[0].fill_(+0.5)
        proj.view_embed.weight[1].fill_(-0.5)

    x = torch.randn(1, 2, 128, 1, 6, 8)
    idx_lr = torch.tensor([[0, 1]], dtype=torch.int64)
    idx_rl = torch.tensor([[1, 0]], dtype=torch.int64)

    with torch.no_grad():
        out_lr = proj(x, idx_lr)
        out_rl = proj(x, idx_rl)

    # Per-view-slot: out_lr[:, 0] should match out_rl[:, 1] under value
    # symmetry (same x, same modality_bias, but view 0 vs view 1 swapped).
    same_view0 = (out_lr[:, 0] - (x[:, 0] + proj.modality_bias.bias.view(1, -1, 1, 1, 1)
                                  + proj.view_embed.weight[0].view(1, -1, 1, 1, 1))).abs().max().item()
    print(f"  out_lr[:, 0] matches manual reconstruction: max diff = {same_view0:.3e}")
    assert same_view0 < 1e-6, f"manual view-0 reconstruction mismatch: {same_view0:.3e}"

    # Position-wise diff between LR and RL must be non-trivial because the
    # view_embed entries are different (+0.5 vs -0.5 above) -- this proves
    # the view routing is doing what we think it does.
    diff = (out_lr - out_rl).abs().max().item()
    print(f"  max|out_lr - out_rl| = {diff:.3e}  (expect >= ~1.0)")
    assert diff >= 0.5, (
        f"swapping view_idx should change output meaningfully; got {diff:.3e}. "
        f"view_embed routing is broken."
    )


# ---------------------------------------------------------------------------
# 4. Gradient flow (alpha bumped so MLP path carries signal)
# ---------------------------------------------------------------------------


def _list_param_names(proj: TactileProjector) -> list[str]:
    return [n for n, _ in proj.named_parameters()]


def _grad_max_abs(p: torch.nn.Parameter) -> float:
    if p.grad is None:
        return -1.0
    return p.grad.abs().max().item()


def _zero_grads(proj: TactileProjector) -> None:
    for p in proj.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def test_gradient_flow():
    """Two-phase gradient flow check (mirrors v0c-A's adapter test).

    Phase A (alpha=0, canonical init):
      AdaLN-Zero invariant. The residual junction ``x + alpha * mlp_out``
      with alpha=0 means d(loss)/d(mlp_out) = upstream * alpha = 0, so every
      param UPSTREAM of the alpha junction (norm, mlp[0], mlp[2]) must have
      ZERO gradient. The DIRECT-path params (alpha, modality_bias.bias,
      view_embed.weight) AND alpha itself must have NON-ZERO gradient (alpha
      via d(loss)/d(alpha) = upstream * mlp_out, which is non-zero because
      mlp[-1] is Kaiming-init).

    Phase B (alpha bumped to 0.1):
      Once alpha leaves zero, the gate opens and ALL params should receive
      non-zero gradient (the full graph is wired).
    """
    _section("4] Two-phase gradient flow (alpha=0 invariant + alpha-bumped)")

    proj = _make_default_projector()
    x = torch.randn(2, 2, 128, 3, 6, 8, requires_grad=False)
    view_idx = _ascending_view_idx(2, 2)

    # ---------- Phase A: canonical init (alpha=0) ----------
    out = proj(x, view_idx)
    loss = out.pow(2).mean()
    loss.backward()

    expected_nonzero_a = ["alpha", "modality_bias.bias", "view_embed.weight"]
    expected_zero_a = [
        "norm.weight", "norm.bias",
        "mlp.0.weight", "mlp.0.bias",
        "mlp.2.weight", "mlp.2.bias",
    ]

    have = dict(proj.named_parameters())
    bad_nonzero: list[str] = []
    for n in expected_nonzero_a:
        gmax = _grad_max_abs(have[n])
        if gmax <= 0:
            bad_nonzero.append(f"{n}: grad |max|={gmax:.3e} (expected > 0)")
    assert not bad_nonzero, (
        "Phase A: required non-zero grads missing:\n  " + "\n  ".join(bad_nonzero)
    )
    bad_zero: list[str] = []
    for n in expected_zero_a:
        gmax = _grad_max_abs(have[n])
        if gmax > 0:
            bad_zero.append(f"{n}: grad |max|={gmax:.3e} (expected exactly 0)")
    assert not bad_zero, (
        "Phase A: AdaLN-Zero invariant violated; norm/mlp params have grad "
        "before alpha leaves zero:\n  " + "\n  ".join(bad_zero)
    )
    print(
        f"  Phase A (alpha=0): "
        f"alpha.grad |max|={_grad_max_abs(have['alpha']):.3e}, "
        f"modality_bias.grad |max|={_grad_max_abs(have['modality_bias.bias']):.3e}, "
        f"view_embed.grad |max|={_grad_max_abs(have['view_embed.weight']):.3e}"
    )
    print(f"  Phase A: norm/mlp params have grad == 0 (AdaLN-Zero invariant) OK")

    # ---------- Phase B: bump alpha and redo ----------
    _zero_grads(proj)
    with torch.no_grad():
        proj.alpha.fill_(0.1)

    out2 = proj(x, view_idx)
    loss2 = out2.pow(2).mean()
    loss2.backward()

    all_required = expected_nonzero_a + expected_zero_a
    bad_b: list[str] = []
    for n in all_required:
        gmax = _grad_max_abs(have[n])
        if gmax <= 0:
            bad_b.append(f"{n}: grad |max|={gmax:.3e} (expected > 0 once alpha != 0)")
    assert not bad_b, (
        "Phase B: alpha bumped but some params still have zero grad -- the "
        "graph is not fully wired:\n  " + "\n  ".join(bad_b)
    )
    print(
        f"  Phase B (alpha=0.1): "
        f"mlp[0].grad |max|={_grad_max_abs(have['mlp.0.weight']):.3e}, "
        f"mlp[2].grad |max|={_grad_max_abs(have['mlp.2.weight']):.3e}, "
        f"norm.grad |max|={_grad_max_abs(have['norm.weight']):.3e}"
    )
    print(f"  Phase B: all {len(all_required)} params have non-zero grad OK")


# ---------------------------------------------------------------------------
# 5. autocast bf16 (CUDA-only)
# ---------------------------------------------------------------------------


def test_autocast_bf16_dtype_and_grads():
    _section("5] autocast bf16: forward succeeds + fp32 master grads")
    if not torch.cuda.is_available():
        print("  CUDA not available -- SKIPPED.")
        return
    if not torch.cuda.is_bf16_supported():
        print("  bf16 not supported on this GPU -- SKIPPED.")
        return

    device = torch.device("cuda")
    proj = _make_default_projector().to(device)
    # Trainable params should remain fp32 master copies (autocast handles
    # the bf16 cast on forward). Verify before forward.
    fp32_params = [n for n, p in proj.named_parameters() if p.dtype == torch.float32]
    n_total = sum(1 for _ in proj.parameters())
    assert len(fp32_params) == n_total, (
        f"expected all {n_total} params in fp32; only {len(fp32_params)} are. "
        f"Trainable bf16 params are footgun-prone in mixed-precision training."
    )
    print(f"  all {n_total} params in fp32 master copies OK")

    with torch.no_grad():
        proj.alpha.fill_(0.1)

    # bf16 input simulates the trainer's data flow under accelerator.bf16.
    x = torch.randn(1, 2, 128, 3, 6, 8, device=device, dtype=torch.bfloat16)
    view_idx = _ascending_view_idx(1, 2).to(device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = proj(x, view_idx)

    # Output dtype: with bf16 input, the residual x + alpha*mlp_out + biases
    # follows bf16 (LayerNorm internally promotes to fp32 then back; the
    # element-wise add inherits the dominant operand's dtype).
    assert out.dtype in (torch.bfloat16, torch.float32), (
        f"under autocast(bf16) with bf16 input, expected bf16 OR fp32 output; "
        f"got {out.dtype}"
    )
    print(f"  autocast(bf16) + bf16 input: out.dtype = {out.dtype} OK")

    # Backward and verify fp32 master grads.
    loss = out.float().pow(2).mean()
    loss.backward()
    bad_dtype = [
        f"{n}: grad.dtype={p.grad.dtype if p.grad is not None else 'None'}"
        for n, p in proj.named_parameters()
        if p.grad is None or p.grad.dtype != torch.float32
    ]
    assert not bad_dtype, f"grads not fp32 (or missing):\n  " + "\n  ".join(bad_dtype)
    print(f"  all backward grads in fp32 OK")

    # Verify an intermediate bf16 cast actually happens. We can only inspect
    # this via a register_hook on the first Linear's output -- inexpensive.
    proj.zero_grad(set_to_none=True)
    intermediate_dtype: list[torch.dtype] = []

    def _grab_dtype(out_t: torch.Tensor) -> None:
        intermediate_dtype.append(out_t.dtype)

    handle = proj.mlp[0].register_forward_hook(
        lambda mod, inp, out: _grab_dtype(out)
    )
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = proj(x, view_idx)
    finally:
        handle.remove()
    assert intermediate_dtype, "forward hook did not fire"
    inter = intermediate_dtype[0]
    assert inter == torch.bfloat16, (
        f"mlp[0] output should be bf16 under autocast(bf16); got {inter}. "
        f"This means autocast is not casting matmuls -- memory savings lost."
    )
    print(f"  mlp[0] intermediate dtype under autocast = {inter} OK")


# ---------------------------------------------------------------------------
# 6. Param count budget
# ---------------------------------------------------------------------------


def test_param_count_under_100k():
    _section("6] Param count < 100K (default 128/256/2)")
    proj = _make_default_projector()
    n = sum(p.numel() for p in proj.parameters())
    breakdown = {
        "norm":          sum(p.numel() for p in proj.norm.parameters()),
        "mlp":           sum(p.numel() for p in proj.mlp.parameters()),
        "alpha":         proj.alpha.numel(),
        "modality_bias": sum(p.numel() for p in proj.modality_bias.parameters()),
        "view_embed":    sum(p.numel() for p in proj.view_embed.parameters()),
    }
    for k, v in breakdown.items():
        print(f"  {k:14s}: {v:>10,}")
    print(f"  {'TOTAL':14s}: {n:>10,}")
    assert n < 100_000, (
        f"projector has {n} params, > 100K budget (Section 3 spec ~66.5K)."
    )


# ---------------------------------------------------------------------------
# 7. num_views=10 forward-compat
# ---------------------------------------------------------------------------


def test_num_views_10_forward_compat():
    _section("7] num_views=10: shape + forward (Stage-3 per-finger ablation)")
    proj = TactileProjector(latent_dim=128, hidden_dim=256, num_views=10)
    proj.eval()
    assert proj.view_embed.weight.shape == (10, 128), (
        f"view_embed.weight.shape = {tuple(proj.view_embed.weight.shape)}; "
        f"expected (10, 128)"
    )
    x = torch.randn(1, 10, 128, 1, 6, 8)
    view_idx = _ascending_view_idx(1, 10)
    with torch.no_grad():
        out = proj(x, view_idx)
    assert out.shape == x.shape, (
        f"V_hand=10 forward shape mismatch: out {tuple(out.shape)} vs "
        f"in {tuple(x.shape)}"
    )
    print(f"  V_hand=10: out.shape = {tuple(out.shape)} OK")
    # Bounds check: passing view_idx >= num_views should raise.
    bad_idx = torch.tensor([[10]], dtype=torch.int64)
    bad_x = torch.randn(1, 1, 128, 1, 6, 8)
    try:
        proj(bad_x, bad_idx)
    except ValueError:
        print("  bounds check: out-of-range view_idx raises ValueError OK")
    else:
        raise AssertionError("expected ValueError for view_idx=10 with num_views=10")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="(no-op; provided for parity with smoke_visual_vae_adapter.py)",
    )
    parser.parse_args()

    print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}")

    test_shape_contract()
    test_init_invariant_residual_is_zero()
    test_view_idx_routes_distinguish_hands()
    test_gradient_flow()
    test_autocast_bf16_dtype_and_grads()
    test_param_count_under_100k()
    test_num_views_10_forward_compat()

    print("\nALL F1 SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
