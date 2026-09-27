"""Local synthetic-tensor smoke for the Stage 1 lite visual VAE adapter.

Mirrors the workflow of :mod:`scripts.smoke_v4_loss`: build the
:class:`models.tactile_models.visual_vae_adapter.VisualVAEAdapterModel` (with
a randomly-initialized LTX VAE so the smoke is CPU-runnable and does not
require the pretrained safetensors), then run a battery of structural and
gradient-flow checks before any training-side change is shipped.

Also runs a YAML dry-run on
``configs/tactile_model/stage1_lite_visual_vae_adapter_v0.yaml`` to confirm
the trainer config schema is consumable by ``VisualVAEAdapterModel`` and that
``runner.visual_vae_adapter_trainer`` imports cleanly (no circular imports).

Checks:

  1. ``GrayToRGB`` initialization produces ``gray.repeat(3)`` exactly when
     ``init_mode='ones_repeat'`` (and ``gray/3`` when ``init_mode='uniform_third'``).
  2. ``FingerAttentionAdapter`` shape contract: any ``(B, 5, C, T, H, W)``
     input -> ``(B, C, T, H, W)`` output with the same spatial layout.
  3. With ``alpha=0`` (zero-init), the adapter output equals the residual
     base (weighted-mean or mean-pool depending on configuration).
  4. With ``finger_logits=0`` (zero-init), the weighted-mean residual reduces
     EXACTLY to a plain mean-pool over fingers within fp tolerance.
  5. The ``finger_embed`` parameter is actually used by attention -- ablating
     it (zeroing the param) changes adapter output (when alpha != 0).
  6. ``AuxFlowDecoder`` (post-fuse) and ``AuxPreFuseFlowHead`` (pre-fuse)
     produce the expected ``(B, F, T_out, 4*H_lat, 4*W_lat, 3)`` shape, with
     ``T_out = 1 + 8*(T_lat - 1)``.
  7. ``AuxPoseDecoder`` produces ``(B, 22)`` from ``(B, C, T, H, W)``.
  8. C5 contract: with full-res input ``(1, 5, 1, 1, 192, 256)`` through the
     wrapper, the post-adapter latent has shape ``(1, 128, T_lat, 6, 8)``.
  9. Frozen-encoder check: every ``vae.parameters()`` entry has
     ``requires_grad=False`` after wrapper construction; the trainable param
     count is broken down by submodule and printed.
  10. Gradient-flow check (the critical one): forward + backward on
      synthetic inputs+targets, then assert
        - ``gray_to_rgb.weight.grad`` is non-None and non-zero,
        - all ``FingerAttentionAdapter`` parameter grads are non-None and non-zero,
        - both ``AuxFlowDecoder`` and ``AuxPreFuseFlowHead`` parameter grads are
          non-None and non-zero,
        - ``AuxPoseDecoder`` parameter grads are non-None and non-zero,
        - every VAE parameter has ``grad is None`` (frozen).
      This is what catches the silent ``torch.no_grad()`` bug where the VAE
      is frozen *and* its computation graph is detached, accidentally
      blocking grads to ``GrayToRGB`` upstream.

CPU is fine -- the LTX VAE forward at small spatial sizes is the dominant
cost. The full-res C5 test is the slowest (~30 s on CPU); other tests are
sub-second. Skip the slow checks with ``--quick``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F


# Make ``models.tactile_models.visual_vae_adapter`` importable.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)


from models.ltx_models.autoencoder_kl_ltx import AutoencoderKLLTXVideo  # noqa: E402
from models.tactile_models.visual_vae_adapter import (  # noqa: E402
    AuxFlowDecoder,
    AuxPoseDecoder,
    AuxPreFuseFlowHead,
    ConcatChannelAdapter,
    FingerAttentionAdapter,
    FingerSetTransformerAdapter,
    GrayToRGB,
    VisualVAEAdapterModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approx_eq(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _make_random_vae() -> AutoencoderKLLTXVideo:
    """Instantiate the LTX VAE with default config (matches the pretrained
    config.json on Delta) BUT without loading weights -- structure is what
    the smoke needs, not numerics.
    """
    return AutoencoderKLLTXVideo()


def _section(title: str) -> None:
    print(f"\n[{title}]")


# ---------------------------------------------------------------------------
# 1. GrayToRGB initialization
# ---------------------------------------------------------------------------


def test_gray_to_rgb_init_ones_repeat():
    _section("1] GrayToRGB ones_repeat init == gray.repeat(3)")
    g = GrayToRGB(init_mode="ones_repeat")
    g.eval()

    # 4-D input.
    gray = torch.randn(2, 1, 8, 12)
    out = g(gray)
    expected = gray.repeat(1, 3, 1, 1)
    assert torch.allclose(out, expected, atol=1e-6), (
        f"4-D ones_repeat mismatch: max diff={(out - expected).abs().max().item()}"
    )
    print("  4-D: max|out - gray.repeat(3)| < 1e-6 OK")

    # 5-D input.
    gray5 = torch.randn(2, 1, 3, 8, 12)
    out5 = g(gray5)
    expected5 = gray5.repeat(1, 3, 1, 1, 1)
    assert torch.allclose(out5, expected5, atol=1e-6), (
        f"5-D ones_repeat mismatch: max diff={(out5 - expected5).abs().max().item()}"
    )
    print("  5-D: max|out - gray.repeat(3)| < 1e-6 OK")

    # Constant input -> constant output (sanity).
    gray_c = torch.full((1, 1, 4, 4), 0.7)
    out_c = g(gray_c)
    assert torch.allclose(out_c, torch.full_like(out_c, 0.7), atol=1e-6)
    print("  constant input: per-channel constant 0.7 OK")


def test_gray_to_rgb_init_uniform_third():
    _section("1b] GrayToRGB uniform_third init == gray / 3 (legacy)")
    g = GrayToRGB(init_mode="uniform_third")
    g.eval()

    gray = torch.randn(2, 1, 4, 4)
    out = g(gray)
    expected = gray.repeat(1, 3, 1, 1) / 3.0
    assert torch.allclose(out, expected, atol=1e-6)
    print("  uniform_third: max|out - gray/3| < 1e-6 OK")


# ---------------------------------------------------------------------------
# 2. FingerAttentionAdapter shape + invariants
# ---------------------------------------------------------------------------


def test_finger_attention_shape():
    _section("2] FingerAttentionAdapter (B, 5, C, T, H, W) -> (B, C, T, H, W)")
    adapter = FingerAttentionAdapter(embed_dim=128, h=6, w=8, num_heads=4)
    z = torch.randn(2, 5, 128, 1, 6, 8)
    out = adapter(z)
    assert out.shape == (2, 128, 1, 6, 8), out.shape
    print(f"  T=1, (6,8): out.shape = {tuple(out.shape)} OK")

    z2 = torch.randn(1, 5, 128, 2, 6, 8)
    out2 = adapter(z2)
    assert out2.shape == (1, 128, 2, 6, 8), out2.shape
    print(f"  T=2, (6,8): out.shape = {tuple(out2.shape)} OK")


def test_finger_attention_alpha_zero_starts_at_residual():
    """At init, alpha=0 so the attention path contributes nothing; output
    must equal the residual base (weighted-mean or mean-pool) exactly.
    """
    _section("3] alpha=0 -> output == residual base")
    torch.manual_seed(0)
    z = torch.randn(2, 5, 128, 2, 6, 8)

    # weighted_mean residual (default).
    adapter = FingerAttentionAdapter(
        embed_dim=128, h=6, w=8, num_heads=4, residual="weighted_mean",
    )
    adapter.eval()
    with torch.no_grad():
        # Sanity: alpha is initialized to zero.
        assert float(adapter.alpha) == 0.0
        out = adapter(z)
        # Residual base with finger_logits=0 (uniform softmax) == mean-pool.
        base = z.mean(dim=1)
    assert torch.allclose(out, base, atol=1e-6), (
        f"weighted_mean alpha=0: max diff = {(out - base).abs().max().item():.3e}"
    )
    print("  weighted_mean: out == z.mean(F) at init OK")

    # Now bump alpha and confirm output diverges -- proves attention is wired.
    with torch.no_grad():
        adapter.alpha.fill_(1.0)
        out_with_attn = adapter(z)
    diff = (out_with_attn - base).abs().max().item()
    assert diff > 1e-3, (
        f"alpha=1.0 should change output noticeably; got max diff = {diff:.3e}"
    )
    print(f"  alpha=1.0: out differs from mean by max {diff:.3e} OK")

    # mean_pool residual variant.
    adapter_mp = FingerAttentionAdapter(
        embed_dim=128, h=6, w=8, num_heads=4, residual="mean_pool",
    )
    adapter_mp.eval()
    with torch.no_grad():
        out_mp = adapter_mp(z)
        base_mp = z.mean(dim=1)
    assert torch.allclose(out_mp, base_mp, atol=1e-6), (
        f"mean_pool alpha=0: max diff = {(out_mp - base_mp).abs().max().item():.3e}"
    )
    print("  mean_pool: out == z.mean(F) at init OK")


def test_finger_attention_finger_logits_zero_equals_mean_pool():
    """The weighted-mean residual base reduces to mean-pool when
    ``finger_logits == 0``. We isolate this via a hand-built computation
    that bypasses attention (alpha=0) so the only thing being tested is the
    residual.
    """
    _section("4] weighted_mean residual (finger_logits=0) == mean_pool")
    torch.manual_seed(0)
    z = torch.randn(3, 5, 128, 1, 6, 8)
    adapter = FingerAttentionAdapter(
        embed_dim=128, h=6, w=8, num_heads=4, residual="weighted_mean",
    )
    adapter.eval()
    with torch.no_grad():
        out = adapter(z)              # alpha=0 -> only residual contributes
        expected = z.mean(dim=1)      # uniform softmax -> mean pool
    diff = (out - expected).abs().max().item()
    assert diff < 1e-6, f"max diff = {diff:.3e}"
    print(f"  max|out - z.mean(F)| = {diff:.3e} OK")

    # Now perturb finger_logits and confirm output diverges from mean.
    with torch.no_grad():
        adapter.finger_logits.copy_(torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0]))
        out2 = adapter(z)
        # Skewed softmax weights.
        w = F.softmax(adapter.finger_logits, dim=0)
        expected2 = (z * w.view(1, 5, 1, 1, 1, 1)).sum(dim=1)
    diff2 = (out2 - expected2).abs().max().item()
    assert diff2 < 1e-6, f"weighted residual mismatch: max diff = {diff2:.3e}"
    diff_vs_mean = (out2 - z.mean(dim=1)).abs().max().item()
    assert diff_vs_mean > 1e-3, (
        f"finger_logits=[3,0,0,0,0] should diverge from mean; "
        f"got {diff_vs_mean:.3e}"
    )
    print(
        f"  finger_logits=[3,0,0,0,0]: diff from mean = {diff_vs_mean:.3e} OK"
    )


def test_finger_attention_finger_embed_used():
    """Ablation: zero-out finger_embed and confirm adapter output changes
    when alpha != 0 (because the only thing finger_embed affects is the
    attention K/V). With alpha=0 it must NOT change (residual ignores
    finger_embed by construction).
    """
    _section("5] finger_embed actually feeds K/V")
    torch.manual_seed(0)
    z = torch.randn(2, 5, 128, 1, 6, 8)
    adapter = FingerAttentionAdapter(
        embed_dim=128, h=6, w=8, num_heads=4, use_finger_embed=True,
    )
    adapter.eval()
    with torch.no_grad():
        adapter.alpha.fill_(1.0)  # silence-> non-silence so attn matters
        out_with = adapter(z)
        finger_embed_backup = adapter.finger_embed.clone()
        adapter.finger_embed.zero_()
        out_zero = adapter(z)
        adapter.finger_embed.copy_(finger_embed_backup)
    diff = (out_with - out_zero).abs().max().item()
    assert diff > 1e-4, (
        f"Zeroing finger_embed should change attn output (alpha=1); "
        f"got max diff = {diff:.3e}"
    )
    print(f"  alpha=1, zero finger_embed: max diff = {diff:.3e} OK")

    # use_finger_embed=False configuration: parameter should not exist.
    adapter_no_fe = FingerAttentionAdapter(
        embed_dim=128, h=6, w=8, num_heads=4, use_finger_embed=False,
    )
    assert adapter_no_fe.finger_embed is None
    out_no_fe = adapter_no_fe(z)
    assert out_no_fe.shape == (2, 128, 1, 6, 8), out_no_fe.shape
    print("  use_finger_embed=False: param absent, fwd still works OK")


def test_finger_attention_pos_query_used():
    """Same idea for pos_embed: zero it and confirm output changes when
    alpha != 0 (pos_embed only feeds the query).
    """
    _section("5b] pos_embed actually feeds the Q")
    torch.manual_seed(0)
    z = torch.randn(1, 5, 128, 1, 6, 8)
    adapter = FingerAttentionAdapter(
        embed_dim=128, h=6, w=8, num_heads=4, use_pos_query=True,
    )
    adapter.eval()
    with torch.no_grad():
        adapter.alpha.fill_(1.0)
        out_with = adapter(z)
        backup = adapter.pos_embed.clone()
        adapter.pos_embed.zero_()
        out_zero = adapter(z)
        adapter.pos_embed.copy_(backup)
    diff = (out_with - out_zero).abs().max().item()
    assert diff > 1e-4, (
        f"Zeroing pos_embed should change attn output (alpha=1); "
        f"got max diff = {diff:.3e}"
    )
    print(f"  alpha=1, zero pos_embed: max diff = {diff:.3e} OK")


# ---------------------------------------------------------------------------
# 6. ConcatChannelAdapter shape (ablation)
# ---------------------------------------------------------------------------


def test_concat_channel_adapter_shape():
    _section("6] ConcatChannelAdapter (ablation) shape")
    adapter = ConcatChannelAdapter(embed_dim=128, num_fingers=5)
    z = torch.randn(2, 5, 128, 1, 6, 8)
    out = adapter(z)
    assert out.shape == (2, 128, 1, 6, 8), out.shape
    print(f"  out.shape = {tuple(out.shape)} OK")


# ---------------------------------------------------------------------------
# 6b. FingerSetTransformerAdapter (v0c-A) shape + invariants
# ---------------------------------------------------------------------------


def test_finger_set_transformer_shape():
    _section("6b.i] FingerSetTransformerAdapter (B,5,C,T,H,W) -> (B,C,T,H,W)")
    adapter = FingerSetTransformerAdapter(
        embed_dim=128, num_fingers=5, num_heads=8,
        num_layers=3, ffn_dim=1024, h=6, w=8,
    )
    for t_lat in (1, 2, 4):
        z = torch.randn(2, 5, 128, t_lat, 6, 8)
        out = adapter(z)
        expected = (2, 128, t_lat, 6, 8)
        assert out.shape == expected, (out.shape, expected, t_lat)
        print(f"  T_lat={t_lat}: out.shape = {tuple(out.shape)} OK")


def test_finger_set_transformer_param_count_dominates_v1():
    _section("6b.ii] FingerSetTransformerAdapter param count >> v1")
    v1 = FingerAttentionAdapter(
        embed_dim=128, num_fingers=5, num_heads=4, h=6, w=8,
    )
    v2 = FingerSetTransformerAdapter(
        embed_dim=128, num_fingers=5, num_heads=8,
        num_layers=3, ffn_dim=1024, h=6, w=8,
    )
    n_v1 = sum(p.numel() for p in v1.parameters())
    n_v2 = sum(p.numel() for p in v2.parameters())
    print(f"  v1 (FingerAttentionAdapter): {n_v1:>10,} params")
    print(f"  v2 (FingerSetTransformer ): {n_v2:>10,} params  (~{n_v2 / max(n_v1, 1):.1f}x v1)")
    assert n_v2 > 5 * n_v1, (
        f"expected v2 to have at least 5x v1 params; got "
        f"v2={n_v2}, v1={n_v1}"
    )


def test_finger_set_transformer_alpha_zero_starts_at_residual():
    """At step 0 (alpha=0), the v2 adapter must produce exactly the same
    output as the v1 adapter with the same hand_query/pos_embed/finger_embed/
    finger_logits values. This guarantees the warm-start invariant: the
    deeper transformer path contributes zero at init, so the model starts
    in function space at the v1 weighted-mean baseline. The transformer
    weights will only start mattering once alpha leaves zero (driven by its
    own gradient, which is non-zero).
    """
    _section("6b.iii] alpha=0 invariant: v2 output equals weighted-mean baseline")
    torch.manual_seed(0)
    v2 = FingerSetTransformerAdapter(
        embed_dim=128, num_fingers=5, num_heads=8,
        num_layers=3, ffn_dim=1024, h=6, w=8,
        residual="weighted_mean",
    )
    v2.eval()
    # Sanity: alpha is exactly 0 at init.
    assert torch.allclose(v2.alpha, torch.zeros_like(v2.alpha)), (
        f"alpha should be zero-init; got {v2.alpha.item()}"
    )
    # Sanity: finger_logits are exactly 0 at init -> uniform softmax ->
    # plain mean over fingers.
    assert torch.allclose(v2.finger_logits, torch.zeros_like(v2.finger_logits))

    z = torch.randn(2, 5, 128, 2, 6, 8)
    with torch.no_grad():
        out = v2(z)
        # Reference: pure mean over fingers (weighted_mean with uniform
        # weights -- equivalent to .mean(dim=1)).
        ref = z.mean(dim=1)
    diff = (out - ref).abs().max().item()
    print(f"  max|out - mean(z, dim=fingers)| = {diff:.3e}")
    assert diff < 1e-5, (
        f"alpha=0 invariant violated: max diff = {diff:.3e} (expected ~0). "
        f"The transformer path must contribute exactly zero at init."
    )


def test_finger_set_transformer_gradient_flow():
    """After a forward + backward with a non-trivial loss, every parameter
    inside the transformer blocks must receive a non-None, non-zero gradient.
    Because alpha=0 at init, the gradient route to the transformer weights
    flows through alpha's own update during one optimizer step OR through
    the second-order effect inside this one backward pass: at alpha=0 the
    transformer-weight gradient IS exactly zero, so we explicitly bump
    alpha to a non-zero value before backward to verify the gradient
    machinery is wired.
    """
    _section("6b.iv] FingerSetTransformerAdapter gradient flow (alpha bumped)")
    torch.manual_seed(0)
    v2 = FingerSetTransformerAdapter(
        embed_dim=128, num_fingers=5, num_heads=8,
        num_layers=3, ffn_dim=1024, h=6, w=8,
    )
    # Bump alpha so the transformer path contributes a non-zero signal,
    # otherwise its weights have grad == 0 by construction at init.
    with torch.no_grad():
        v2.alpha.fill_(0.1)
    z = torch.randn(2, 5, 128, 2, 6, 8, requires_grad=False)
    out = v2(z)
    loss = out.pow(2).mean()
    loss.backward()

    # alpha must have a non-zero grad even at init (it always does because
    # base + alpha*hand_out depends linearly on alpha through hand_out).
    assert v2.alpha.grad is not None and v2.alpha.grad.abs().max().item() > 0, (
        f"alpha.grad should be non-zero; got {v2.alpha.grad}"
    )

    # All transformer-block weights must have non-None, non-zero grad now
    # that alpha != 0.
    bad: List[str] = []
    for n, p in v2.named_parameters():
        if not n.startswith("blocks."):
            continue
        if p.grad is None:
            bad.append(f"{n}: grad is None")
            continue
        if p.grad.abs().max().item() == 0:
            bad.append(f"{n}: grad is all zeros")
    assert not bad, (
        "transformer blocks have missing / zero gradients after backward "
        "(alpha was bumped to 0.1):\n  " + "\n  ".join(bad)
    )
    print(f"  alpha.grad = {v2.alpha.grad.abs().max().item():.3e}, "
          f"all {sum(1 for n, _ in v2.named_parameters() if n.startswith('blocks.'))} "
          f"transformer-block params have non-zero grad OK")


def test_wrapper_finger_set_transformer_e2e():
    """End-to-end on a tiny synthetic batch with adapter_kind=
    'finger_set_transformer'. Mirrors test 9 (small-res forward) but with the
    new adapter so we catch any wiring bug between the wrapper and the new
    adapter class before launching a multi-hour training run.
    """
    _section("6b.v] VisualVAEAdapterModel(adapter_kind='finger_set_transformer') E2E")
    torch.manual_seed(0)
    model = VisualVAEAdapterModel(
        vae=_make_random_vae(),
        latent_channels=128,
        adapter_kind="finger_set_transformer",
        num_fingers=5,
        num_heads=8,
        spatial_h=6, spatial_w=8,
        adapter_n_layers=3,
        adapter_ffn_dim=1024,
        enable_pre_fuse=True,
    )
    model.eval()
    # Frozen-VAE invariant.
    n_vae_train = sum(p.numel() for p in model.vae.parameters() if p.requires_grad)
    assert n_vae_train == 0, (
        f"frozen-VAE invariant violated: {n_vae_train} VAE params with "
        f"requires_grad=True after construction with adapter_kind='finger_set_transformer'"
    )
    # Adapter type sanity.
    assert isinstance(model.adapter, FingerSetTransformerAdapter), (
        f"expected FingerSetTransformerAdapter; got {type(model.adapter)}"
    )
    # Tiny forward (T=1, low spatial to keep the random VAE fast).
    tactile = torch.randn(1, 5, 1, 1, 192, 256)
    with torch.no_grad():
        out = model(tactile)
    assert out["flow_pred_post"] is not None
    assert out["pose_pred"] is not None
    assert out["z_per_hand"].shape[1:] == (128, 1, 6, 8), (
        f"C5 contract violated: z_per_hand.shape = {tuple(out['z_per_hand'].shape)}"
    )
    print(f"  E2E forward OK; z_per_hand shape = {tuple(out['z_per_hand'].shape)}")


# ---------------------------------------------------------------------------
# 6c. v0d additions: pose injection + TimeSformer post-adapter + dropped
# AuxPoseDecoder (Option A). Default kwargs reduce v0d to v0c-A so all
# alpha-zero invariants below also serve as warm-start safety nets.
# ---------------------------------------------------------------------------


def _build_v0c_a_adapter() -> FingerSetTransformerAdapter:
    """Reference v0c-A adapter: all v0d flags off, defaults match the
    v0c-A_FULL yaml.
    """
    return FingerSetTransformerAdapter(
        embed_dim=128, num_fingers=5, num_heads=8,
        num_layers=3, ffn_dim=1024, h=6, w=8,
        residual="weighted_mean",
    )


def _build_v0d_adapter(
    use_pose_injection: bool = True,
    use_timesformer: bool = True,
    finger_dropout: float = 0.0,
) -> FingerSetTransformerAdapter:
    """Reference v0d adapter: by default the three TouchAnything-aligned
    upgrades are ON; the individual sub-tests below override one flag at
    a time.
    """
    return FingerSetTransformerAdapter(
        embed_dim=128, num_fingers=5, num_heads=8,
        num_layers=3, ffn_dim=1024, h=6, w=8,
        residual="weighted_mean",
        use_pose_injection=use_pose_injection,
        pose_dim=22,
        use_timesformer=use_timesformer,
        timesformer_num_blocks=2,
        timesformer_num_heads=8,
        timesformer_ffn_dim=512,
        finger_dropout=finger_dropout,
    )


def _copy_shared_weights(src: FingerSetTransformerAdapter, dst: FingerSetTransformerAdapter) -> None:
    """Copy the v0c-A weight subset (everything except v0d-only keys) from
    ``src`` into ``dst``. Used by the alpha-zero invariant sub-tests so the
    two adapters share random init exactly on the shared weights and only
    differ in the v0d additions.
    """
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    # Filter to keys that exist in both (i.e. drop v0d-only keys).
    shared = {k: v.clone() for k, v in src_sd.items() if k in dst_sd}
    missing, unexpected = dst.load_state_dict(shared, strict=False)
    # Sanity: only v0d-new keys are missing; nothing unexpected.
    assert not unexpected, (
        f"_copy_shared_weights: unexpected keys leaked into dst: {unexpected}"
    )
    # `missing` are the v0d-only keys (pose_encoder.*, timesformer_blocks.*,
    # alpha_pose, alpha_temp, mask_token). We don't assert the exact set here
    # because mask_token is only present when finger_dropout > 0.


def test_v0d_param_count():
    """6c.i] v0d adds ~1M params over v0c-A.

    Per the plan's Section 4 budget:
        v0c-A : ~1.6M (3-layer transformer)
        v0d   : ~2.6M = v0c-A + 21K (_PoseEncoder) + 2 x ~510K (_TimeSformerBlock)
                       + 2 (alpha_pose, alpha_temp) + 0 (mask_token off)
        Delta : ~+1.0M
    """
    _section("6c.i] v0d param count = v0c-A + pose + timesformer (~+1.0M)")
    v0c = _build_v0c_a_adapter()
    v0d = _build_v0d_adapter()  # all flags on, dropout off
    n_v0c = sum(p.numel() for p in v0c.parameters())
    n_v0d = sum(p.numel() for p in v0d.parameters())
    delta = n_v0d - n_v0c
    print(f"  v0c-A  : {n_v0c:>10,} params")
    print(f"  v0d    : {n_v0d:>10,} params  (+{delta:,} = {delta/1e6:.2f} M)")
    # Pose encoder is ~21k (Linear(22->128)+LN+GELU+Linear(128->128)+LN ~= 24k).
    # Each TimeSformer block is ~510k (2x MHA + 4x FFN + 3x LN at 128-d).
    # Total v0d delta should land in ~0.5M..1.5M -- assert the loose band so
    # the test stays stable across minor hidden-dim adjustments.
    assert 0.5e6 <= delta <= 1.5e6, (
        f"v0d delta {delta/1e6:.2f} M outside [0.5, 1.5] M expected band. "
        f"If the v0d hyperparams changed deliberately, update this band."
    )
    # alpha_pose + alpha_temp = 2 scalar gates.
    n_alphas = (
        v0d.alpha_pose.numel()
        + v0d.alpha_temp.numel()
    )
    assert n_alphas == 2, n_alphas
    print(f"  alpha gates: alpha_pose={v0d.alpha_pose.numel()}  "
          f"alpha_temp={v0d.alpha_temp.numel()}  OK")


def test_v0d_alpha_pose_zero_equals_v0c_a():
    """6c.ii] use_pose_injection=True + alpha_pose=0 -> bit-equal v0c-A.

    Builds two adapters with shared v0c-A weights; the v0d variant has
    use_pose_injection=True but alpha_pose=0 (the __init__ default). Output
    must equal the v0c-A baseline (which is z.mean(dim=fingers) because
    finger_logits=0 too).
    """
    _section("6c.ii] alpha_pose=0 invariant: v0d output == v0c-A baseline")
    torch.manual_seed(0)
    v0c = _build_v0c_a_adapter()
    v0d = _build_v0d_adapter(use_pose_injection=True, use_timesformer=False)
    _copy_shared_weights(v0c, v0d)

    # Sanity: alpha and alpha_pose are zero at init.
    assert float(v0d.alpha) == 0.0
    assert float(v0d.alpha_pose) == 0.0

    v0c.eval(); v0d.eval()
    z = torch.randn(2, 5, 128, 2, 6, 8)
    pose = torch.randn(2, 2, 22)
    with torch.no_grad():
        out_v0c = v0c(z)
        out_v0d = v0d(z, hand_pose=pose)
    diff = (out_v0c - out_v0d).abs().max().item()
    print(f"  max|v0d - v0c-A| = {diff:.3e}")
    assert diff < 1e-5, (
        f"alpha_pose=0 invariant violated: max diff = {diff:.3e}. "
        f"The pose-injection path must contribute exactly 0 at init."
    )


def test_v0d_alpha_temp_zero_equals_v0c_a():
    """6c.iii] use_timesformer=True + alpha_temp=0 -> bit-equal v0c-A."""
    _section("6c.iii] alpha_temp=0 invariant: v0d output == v0c-A baseline")
    torch.manual_seed(0)
    v0c = _build_v0c_a_adapter()
    v0d = _build_v0d_adapter(use_pose_injection=False, use_timesformer=True)
    _copy_shared_weights(v0c, v0d)

    assert float(v0d.alpha) == 0.0
    assert float(v0d.alpha_temp) == 0.0

    v0c.eval(); v0d.eval()
    z = torch.randn(2, 5, 128, 2, 6, 8)
    with torch.no_grad():
        out_v0c = v0c(z)
        out_v0d = v0d(z)  # use_pose_injection=False -> hand_pose silently ignored
    diff = (out_v0c - out_v0d).abs().max().item()
    print(f"  max|v0d - v0c-A| = {diff:.3e}")
    assert diff < 1e-5, (
        f"alpha_temp=0 invariant violated: max diff = {diff:.3e}. "
        f"The TimeSformer post-adapter path must contribute exactly 0 at init."
    )


def test_v0d_both_alphas_zero_equals_v0c_a():
    """6c.iv] both flags ON + both alphas=0 -> bit-equal v0c-A.

    This is the warm-start safety net: v0d-with-flags-on, fresh-init, eval
    mode (so finger_dropout is a no-op even if enabled) must produce the
    exact same output as v0c-A. If this passes, loading a v0c-A ckpt into
    a v0d model with strict=False and starting training is guaranteed to
    not destabilize the model on step 0.
    """
    _section("6c.iv] both alphas=0 invariant: v0d (all flags on) == v0c-A")
    torch.manual_seed(0)
    v0c = _build_v0c_a_adapter()
    v0d = _build_v0d_adapter(use_pose_injection=True, use_timesformer=True)
    _copy_shared_weights(v0c, v0d)

    assert float(v0d.alpha) == 0.0
    assert float(v0d.alpha_pose) == 0.0
    assert float(v0d.alpha_temp) == 0.0

    v0c.eval(); v0d.eval()
    z = torch.randn(2, 5, 128, 2, 6, 8)
    pose = torch.randn(2, 2, 22)
    with torch.no_grad():
        out_v0c = v0c(z)
        out_v0d = v0d(z, hand_pose=pose)
    diff = (out_v0c - out_v0d).abs().max().item()
    print(f"  max|v0d - v0c-A| = {diff:.3e}")
    assert diff < 1e-5, (
        f"both-alphas-zero invariant violated: max diff = {diff:.3e}. "
        f"Warm-start from v0c-A ckpt is NOT safe -- the v0d additions are "
        f"leaking signal at init. Check alpha_pose / alpha_temp init."
    )


def test_v0d_gradient_flow_after_alpha_bumps():
    """6c.v] After bumping alphas off zero, gradient reaches all new params.

    Mirror of v0c-A test 6b.iv: alpha=0 silences the relevant path by
    construction, so we bump alpha_pose=alpha_temp=0.1 then verify
    pose_encoder.* and timesformer_blocks.* receive non-zero grads.
    alpha_pose / alpha_temp themselves always receive grad because the
    output depends linearly on them through their residual.
    """
    _section("6c.v] v0d gradient flow (alphas bumped to 0.1)")
    torch.manual_seed(0)
    v0d = _build_v0d_adapter(
        use_pose_injection=True,
        use_timesformer=True,
        finger_dropout=0.5,  # exercise the mask_token path too
    )
    v0d.train()
    with torch.no_grad():
        v0d.alpha.fill_(0.1)
        v0d.alpha_pose.fill_(0.1)
        v0d.alpha_temp.fill_(0.1)

    z = torch.randn(2, 5, 128, 2, 6, 8, requires_grad=False)
    pose = torch.randn(2, 2, 22, requires_grad=False)
    out = v0d(z, hand_pose=pose)
    loss = out.pow(2).mean()
    loss.backward()

    # alpha_pose / alpha_temp must have non-zero grad (linear residual).
    assert v0d.alpha_pose.grad is not None and v0d.alpha_pose.grad.abs().max().item() > 0, (
        f"alpha_pose.grad should be non-zero; got {v0d.alpha_pose.grad}"
    )
    assert v0d.alpha_temp.grad is not None and v0d.alpha_temp.grad.abs().max().item() > 0, (
        f"alpha_temp.grad should be non-zero; got {v0d.alpha_temp.grad}"
    )

    # All pose_encoder params must have non-zero grad now that alpha_pose != 0.
    bad: List[str] = []
    for n, p in v0d.named_parameters():
        if n.startswith("pose_encoder."):
            if p.grad is None or p.grad.abs().max().item() == 0:
                bad.append(f"pose_encoder.{n}: "
                           f"{'None' if p.grad is None else 'all zero'}")
    assert not bad, (
        "pose_encoder has missing / zero grads after backward (alpha_pose=0.1):\n"
        "  " + "\n  ".join(bad)
    )

    # All timesformer_blocks params must have non-zero grad.
    bad = []
    for n, p in v0d.named_parameters():
        if n.startswith("timesformer_blocks."):
            if p.grad is None or p.grad.abs().max().item() == 0:
                bad.append(f"{n}: "
                           f"{'None' if p.grad is None else 'all zero'}")
    assert not bad, (
        "timesformer_blocks has missing / zero grads after backward "
        "(alpha_temp=0.1):\n  " + "\n  ".join(bad)
    )

    # mask_token must have non-zero grad (finger_dropout=0.5 + training)
    assert v0d.mask_token is not None
    assert v0d.mask_token.grad is not None and v0d.mask_token.grad.abs().max().item() > 0, (
        f"mask_token.grad should be non-zero in training with finger_dropout=0.5; "
        f"got {v0d.mask_token.grad}"
    )
    print(
        f"  alpha_pose.grad={v0d.alpha_pose.grad.abs().max().item():.3e}, "
        f"alpha_temp.grad={v0d.alpha_temp.grad.abs().max().item():.3e}, "
        f"pose_encoder/timesformer/mask_token all have non-zero grad OK"
    )


def test_v0d_wrapper_drops_aux_pose():
    """6c.vi] Wrapper Option A: use_pose_injection=True -> aux_pose is None.

    Mirrors TouchAnything's tactile_prediction branch (no PoseDecoder).
    Trainer gates loss_pose on `out["pose_pred"] is None`.
    """
    _section("6c.vi] VisualVAEAdapterModel: aux_pose dropped when pose injected")
    torch.manual_seed(0)
    model = VisualVAEAdapterModel(
        vae=_make_random_vae(),
        latent_channels=128,
        adapter_kind="finger_set_transformer",
        spatial_h=2, spatial_w=2,
        enable_pre_fuse=False,
        adapter_use_pose_injection=True,
        adapter_pose_dim=22,
    )
    assert model.aux_pose is None, (
        f"aux_pose must be None when use_pose_injection=True (Option A); "
        f"got {model.aux_pose!r}"
    )
    assert model.use_pose_injection is True
    print("  use_pose_injection=True -> model.aux_pose is None OK")

    # Forward path: pose_pred should be None.
    model.eval()
    tactile = torch.randn(1, 5, 1, 1, 64, 64).clamp(-1, 1)
    pose = torch.randn(1, 1, 22)
    with torch.no_grad():
        out = model(tactile, hand_pose=pose)
    assert out["pose_pred"] is None, (
        f"pose_pred should be None when use_pose_injection=True; "
        f"got tensor of shape {tuple(out['pose_pred'].shape)}"
    )
    print("  forward(tactile, hand_pose) -> out['pose_pred'] is None OK")


def test_v0d_wrapper_rejects_v0d_kwargs_on_other_adapters():
    """6c.vii] v0d kwargs on a non-finger_set_transformer adapter -> ValueError.

    Silent-drop would let a user enable pose injection in YAML with the v0
    adapter and never notice the kwarg has no effect.
    """
    _section("6c.vii] v0d kwargs gate: ValueError on non-finger_set_transformer")
    try:
        VisualVAEAdapterModel(
            vae=_make_random_vae(),
            latent_channels=128,
            adapter_kind="finger_attention",          # v0
            spatial_h=2, spatial_w=2,
            adapter_use_pose_injection=True,          # v0d-only
        )
    except ValueError as e:
        print(f"  finger_attention + use_pose_injection -> ValueError OK: {str(e)[:80]}...")
    else:
        raise AssertionError(
            "expected ValueError when passing v0d kwargs to a non-finger_set_transformer "
            "adapter_kind, got silent acceptance"
        )

    # use_timesformer=True
    try:
        VisualVAEAdapterModel(
            vae=_make_random_vae(),
            latent_channels=128,
            adapter_kind="concat_channel",
            spatial_h=2, spatial_w=2,
            adapter_use_timesformer=True,
        )
    except ValueError as e:
        print(f"  concat_channel + use_timesformer -> ValueError OK: {str(e)[:80]}...")
    else:
        raise AssertionError(
            "expected ValueError when passing use_timesformer=True to concat_channel adapter"
        )

    # finger_dropout > 0
    try:
        VisualVAEAdapterModel(
            vae=_make_random_vae(),
            latent_channels=128,
            adapter_kind="finger_attention",
            spatial_h=2, spatial_w=2,
            adapter_finger_dropout=0.1,
        )
    except ValueError as e:
        print(f"  finger_attention + finger_dropout -> ValueError OK: {str(e)[:80]}...")
    else:
        raise AssertionError(
            "expected ValueError when passing finger_dropout > 0 to finger_attention adapter"
        )


def test_v0d_align_pose_to_lat():
    """6c.viii] _align_pose_to_lat: T_raw=1 broadcast, T_raw=T_lat identity,
    T=9 -> T_lat=2 linspace picks indices [0, 8] (strict superset of v0c-A).
    """
    _section("6c.viii] VisualVAEAdapterModel._align_pose_to_lat resampling")

    # T_raw == T_lat: identity.
    pose = torch.randn(2, 3, 22)
    out = VisualVAEAdapterModel._align_pose_to_lat(pose, T_lat=3)
    assert torch.equal(out, pose), "T_raw == T_lat should be identity"
    print(f"  T_raw=3 T_lat=3: identity OK (no copy)")

    # T_raw=1, T_lat=2: broadcast.
    pose1 = torch.randn(2, 1, 22)
    out1 = VisualVAEAdapterModel._align_pose_to_lat(pose1, T_lat=2)
    assert out1.shape == (2, 2, 22), out1.shape
    assert torch.equal(out1[:, 0], pose1[:, 0])
    assert torch.equal(out1[:, 1], pose1[:, 0])
    print(f"  T_raw=1 T_lat=2: broadcast {tuple(pose1.shape)} -> {tuple(out1.shape)} OK")

    # T_raw=9, T_lat=2: indices [0, 8] -- slot 1 == frame 8 == v0c-A last_frame
    # pose. THIS is the strict-superset claim from plan Section 6.3.
    pose9 = torch.arange(2 * 9 * 22, dtype=torch.float32).reshape(2, 9, 22)
    out9 = VisualVAEAdapterModel._align_pose_to_lat(pose9, T_lat=2)
    assert out9.shape == (2, 2, 22), out9.shape
    assert torch.equal(out9[:, 0], pose9[:, 0]), \
        "slot 0 must match clip-start frame 0"
    assert torch.equal(out9[:, 1], pose9[:, 8]), \
        "slot 1 must match clip-end frame 8 (v0c-A last_frame pose ground truth)"
    print(f"  T_raw=9 T_lat=2: indices [0, 8] (slot 1 == frame 8 == v0c-A last_frame) OK")

    # T_raw=17, T_lat=3: indices [0, 8, 16] -- ditto.
    pose17 = torch.arange(1 * 17 * 22, dtype=torch.float32).reshape(1, 17, 22)
    out17 = VisualVAEAdapterModel._align_pose_to_lat(pose17, T_lat=3)
    assert out17.shape == (1, 3, 22)
    # round(linspace(0, 16, 3)) = [0, 8, 16]
    for slot, idx in [(0, 0), (1, 8), (2, 16)]:
        assert torch.equal(out17[:, slot], pose17[:, idx]), (slot, idx)
    print(f"  T_raw=17 T_lat=3: indices [0, 8, 16] OK")


# ---------------------------------------------------------------------------
# 7. Aux head shapes
# ---------------------------------------------------------------------------


def _expected_t_out(t_lat: int) -> int:
    """LTX-style 8x temporal upsample: T_out = 1 + 8*(T_lat - 1)."""
    return 1 + 8 * (t_lat - 1) if t_lat > 1 else 1


def test_aux_pre_fuse_head_shape():
    _section("7a] AuxPreFuseFlowHead shape")
    head = AuxPreFuseFlowHead(latent_channels=128, num_fingers=5)
    head.eval()
    for t_lat in (1, 2):
        z = torch.randn(1, 5, 128, t_lat, 6, 8)
        with torch.no_grad():
            flow = head(z)
        t_out = _expected_t_out(t_lat)
        expected = (1, 5, t_out, 24, 32, 3)
        assert flow.shape == expected, f"T_lat={t_lat}: {flow.shape} vs {expected}"
        print(f"  T_lat={t_lat}: out.shape = {tuple(flow.shape)} OK")


def test_aux_flow_decoder_shape():
    _section("7b] AuxFlowDecoder shape")
    head = AuxFlowDecoder(latent_channels=128, num_fingers=5)
    head.eval()
    for t_lat in (1, 2):
        z = torch.randn(1, 128, t_lat, 6, 8)
        with torch.no_grad():
            flow = head(z)
        t_out = _expected_t_out(t_lat)
        expected = (1, 5, t_out, 24, 32, 3)
        assert flow.shape == expected, f"T_lat={t_lat}: {flow.shape} vs {expected}"
        print(f"  T_lat={t_lat}: out.shape = {tuple(flow.shape)} OK")


def test_aux_pose_decoder_shape():
    _section("7c] AuxPoseDecoder shape")
    head = AuxPoseDecoder(latent_channels=128)
    head.eval()
    z = torch.randn(3, 128, 2, 6, 8)
    with torch.no_grad():
        pose = head(z)
    assert pose.shape == (3, 22), pose.shape
    print(f"  out.shape = {tuple(pose.shape)} OK")


# ---------------------------------------------------------------------------
# 8. Wrapper structural checks (frozen VAE, param breakdown)
# ---------------------------------------------------------------------------


def _build_wrapper() -> VisualVAEAdapterModel:
    return VisualVAEAdapterModel(
        vae=_make_random_vae(),
        latent_channels=128,
        adapter_kind="finger_attention",
        num_fingers=5,
        num_heads=4,
        spatial_h=6,
        spatial_w=8,
        use_finger_embed=True,
        use_pos_query=True,
        adapter_residual="weighted_mean",
        gray_to_rgb_init="ones_repeat",
        latent_mode="mean",
        enable_pre_fuse=True,
    )


def test_wrapper_frozen_vae_and_param_breakdown():
    _section("8] Wrapper: frozen VAE + trainable param breakdown")
    model = _build_wrapper()

    # All VAE params must be frozen.
    n_vae = 0
    for p in model.vae.parameters():
        assert not p.requires_grad, "frozen VAE should not require grad"
        n_vae += p.numel()
    print(f"  vae params (frozen) : {n_vae/1e6:7.2f} M")

    # Per-submodule trainable counts.
    def count_trainable(mod) -> int:
        return sum(p.numel() for p in mod.parameters() if p.requires_grad)

    breakdown = {
        "gray_to_rgb": count_trainable(model.gray_to_rgb),
        "adapter": count_trainable(model.adapter),
        "modality_embed": count_trainable(model.modality_embed),
        "aux_flow_post": count_trainable(model.aux_flow_post),
        "aux_pose": count_trainable(model.aux_pose),
        "aux_flow_pre": (
            count_trainable(model.aux_flow_pre) if model.aux_flow_pre is not None else 0
        ),
    }
    n_train = sum(breakdown.values())
    for k, v in breakdown.items():
        print(f"  {k:14s}: {v/1e6:7.3f} M")
    print(f"  TOTAL trainable: {n_train/1e6:7.3f} M")

    # Sanity: trainable should be much smaller than VAE.
    assert n_train < n_vae, (
        f"trainable ({n_train}) should be <<< vae ({n_vae})"
    )
    # The plan estimated ~1.5M, but the two aux flow heads dominate at
    # ~2.7M each (LTXVideoResnetBlock3d with k=3 has ~220k params per
    # block, and each upsample stack uses 4-5 of those). Total trainable
    # ~5.5M is still <2% of the frozen LTX VAE (~419M), which is the
    # actual constraint that matters for "small adapter on top of frozen
    # encoder". The 8M cap below leaves headroom for the optional
    # QFormer ablation later without inflating without bound.
    assert n_train <= 8_000_000, f"too many trainable params: {n_train}"
    print(f"  trainable << frozen-VAE ({n_train/n_vae*100:.2f}%) OK")


# ---------------------------------------------------------------------------
# 9. Wrapper forward shape sanity (small-res, fast)
# ---------------------------------------------------------------------------


def test_wrapper_forward_small():
    """Forward at a small spatial size (still 32x-divisible) so the smoke
    is fast enough on CPU. The C5 contract is checked with full res in a
    separate test guarded by --quick.
    """
    _section("9] Wrapper forward (small-res 64x64)")
    torch.manual_seed(0)
    model = _build_wrapper()
    model.eval()

    # 64 / 32 = 2, so latent grid is (2, 2). Build an adapter with the
    # matching small spatial config -- need a *fresh* wrapper for that.
    model_small = VisualVAEAdapterModel(
        vae=_make_random_vae(),
        latent_channels=128,
        spatial_h=2,
        spatial_w=2,
        adapter_kind="finger_attention",
    )
    model_small.eval()

    tactile = torch.randn(1, 5, 1, 1, 64, 64).clamp(-1, 1)
    with torch.no_grad():
        out = model_small(tactile)
    assert out["z_per_finger"].shape == (1, 5, 128, 1, 2, 2), out["z_per_finger"].shape
    assert out["z_per_hand"].shape == (1, 128, 1, 2, 2), out["z_per_hand"].shape
    assert out["pose_pred"].shape == (1, 22), out["pose_pred"].shape
    # Aux heads upsample 4x spatially -> (8, 8).
    assert out["flow_pred_post"].shape == (1, 5, 1, 8, 8, 3), out["flow_pred_post"].shape
    assert out["flow_pred_pre"].shape == (1, 5, 1, 8, 8, 3), out["flow_pred_pre"].shape
    print("  z_per_finger    :", tuple(out["z_per_finger"].shape))
    print("  z_per_hand      :", tuple(out["z_per_hand"].shape))
    print("  flow_pred_post  :", tuple(out["flow_pred_post"].shape))
    print("  flow_pred_pre   :", tuple(out["flow_pred_pre"].shape))
    print("  pose_pred       :", tuple(out["pose_pred"].shape))


# ---------------------------------------------------------------------------
# 10. C5 contract (full-res 192x256 -> 6x8 latent). SLOW.
# ---------------------------------------------------------------------------


def test_wrapper_c5_contract_full_res():
    _section("10] C5 contract: 192x256 input -> (6, 8) latent (SLOW)")
    torch.manual_seed(0)
    model = _build_wrapper()
    model.eval()

    tactile = torch.randn(1, 5, 1, 1, 192, 256).clamp(-1, 1)
    with torch.no_grad():
        out = model(tactile)

    # C5 contract: post-adapter latent must be (B, 128, T_lat, 6, 8).
    z_finger = out["z_per_finger"]
    z_hand = out["z_per_hand"]
    assert z_finger.shape == (1, 5, 128, 1, 6, 8), z_finger.shape
    assert z_hand.shape == (1, 128, 1, 6, 8), z_hand.shape
    # The adapter must NOT change the spatial / temporal dims.
    assert z_finger.shape[2:] == (128, 1, 6, 8)  # (C, T, H, W) per finger
    assert z_finger.shape[3:] == z_hand.shape[2:], (
        f"C5 violation: z_finger spatial-temporal layout {z_finger.shape[3:]} "
        f"must equal z_hand {z_hand.shape[2:]}"
    )
    print("  z_per_finger:", tuple(z_finger.shape))
    print("  z_per_hand  :", tuple(z_hand.shape))
    print("  flow_post   :", tuple(out["flow_pred_post"].shape))
    print("  flow_pre    :", tuple(out["flow_pred_pre"].shape))
    print("  pose        :", tuple(out["pose_pred"].shape))
    # Auxiliary flow heads upsample 4x to (24, 32) = the v3/v4 GT layout.
    assert out["flow_pred_post"].shape == (1, 5, 1, 24, 32, 3)
    assert out["flow_pred_pre"].shape == (1, 5, 1, 24, 32, 3)
    print("  C5 spatial layout matches visual VAE single-view OK")


# ---------------------------------------------------------------------------
# 11. Gradient-flow check (the critical one)
# ---------------------------------------------------------------------------


def _fake_targets(out_post: torch.Tensor, pose_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Synthetic GT with the same shape so we can compute MSE losses."""
    return torch.zeros_like(out_post), torch.zeros_like(pose_pred)


def _build_small_wrapper_with_pre() -> VisualVAEAdapterModel:
    return VisualVAEAdapterModel(
        vae=_make_random_vae(),
        latent_channels=128,
        spatial_h=2,
        spatial_w=2,
        adapter_kind="finger_attention",
        enable_pre_fuse=True,
    )


def _zero_grads(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def test_wrapper_gradient_flow():
    """The most important test in this smoke. After ``loss.backward()``:

      * gradients must reach ``GrayToRGB`` (otherwise the input-side adapter
        cannot learn anything) AND every other trainable submodule on a
        non-attention path;
      * the AdaLN-Zero invariant for the attention path is verified
        separately by bumping alpha off zero and confirming the attention
        params then receive grad;
      * VAE parameters must have ``grad is None`` (frozen, not just zero).

    This catches the silent bug where wrapping ``vae.encode`` in
    ``torch.no_grad()`` would block grads to ``GrayToRGB`` even though
    ``vae.parameters()`` correctly have ``requires_grad=False``.

    NOTE on alpha=0 init:
        At init, ``alpha=0`` so ``out = base + 0 * attn_out``. Therefore
        ``d(loss)/d(attn_out) = 0`` and every parameter UPSTREAM of
        attention (``hand_query``, ``pos_embed``, ``finger_embed``, the
        attn linear weights, the q/kv layer norms) receives zero grad at
        the very first step. This is the intended AdaLN-Zero behavior:
        the residual base learns first, then alpha grows, then the
        attention path opens up. ``alpha`` itself and the residual-base
        params (``finger_logits`` for weighted-mean) DO receive grad at
        init -- they are tested explicitly.
    """
    # ------------------------------------------------------------------
    # Phase A: alpha=0 init -> grads reach gray_to_rgb, residual base,
    #          aux heads, modality_embed, VAE has no grad.
    # ------------------------------------------------------------------
    _section("11] Gradient flow (alpha=0 init): residual + aux paths")
    torch.manual_seed(0)
    model = _build_small_wrapper_with_pre()
    model.train()

    tactile = torch.randn(1, 5, 1, 1, 64, 64).clamp(-1, 1)
    out = model(tactile)

    flow_target_post, pose_target = _fake_targets(out["flow_pred_post"], out["pose_pred"])
    flow_target_pre = torch.zeros_like(out["flow_pred_pre"])

    loss = (
        F.mse_loss(out["flow_pred_post"], flow_target_post)
        + 0.3 * F.mse_loss(out["flow_pred_pre"], flow_target_pre)
        + 0.1 * F.mse_loss(out["pose_pred"], pose_target)
    )
    loss.backward()
    print(f"  loss (random init, alpha=0): {float(loss):.6f}")

    # 1. GrayToRGB must receive non-zero grad. This is THE check that
    #    catches the silent torch.no_grad() bug around vae.encode.
    g_w = model.gray_to_rgb.conv.weight.grad
    g_b = model.gray_to_rgb.conv.bias.grad
    assert g_w is not None, "gray_to_rgb.conv.weight.grad is None -- grads not flowing"
    assert g_b is not None, "gray_to_rgb.conv.bias.grad is None"
    g_w_abs = g_w.abs().sum().item()
    g_b_abs = g_b.abs().sum().item()
    assert g_w_abs > 0.0, f"gray_to_rgb.weight grad sum = {g_w_abs} (must be > 0)"
    print(
        f"  gray_to_rgb.weight grad |sum|={g_w_abs:.3e}, "
        f"bias |sum|={g_b_abs:.3e}: GRADS REACHED OK"
    )

    # 2. At init, the residual path (alpha + finger_logits) AND the path
    #    `gray_to_rgb -> vae -> z_kv -> base -> ...` must have grad. The
    #    attention path (hand_query, pos_embed, finger_embed, attn weights,
    #    q/kv norm) is silenced by alpha=0 -- this is the expected
    #    AdaLN-Zero behavior. We allow zero grad on those at init but
    #    require non-None (the autograd graph must still be built).
    residual_path_params = {"alpha", "finger_logits"}
    attn_path_param_prefixes = (
        "hand_query", "pos_embed", "finger_embed",
        "q_norm", "kv_norm", "attn",
    )
    for name, p in model.adapter.named_parameters():
        assert p.grad is not None, f"adapter.{name} has grad=None"
        s = p.grad.abs().sum().item()
        if name in residual_path_params:
            assert s > 0.0, (
                f"adapter.{name} (residual path) grad sum = {s} -- expected > 0"
            )
        else:
            # Attention-path param: by AdaLN-Zero design grad is zero at
            # init. We just sanity-check that it's not None.
            assert any(name.startswith(p_) for p_ in attn_path_param_prefixes), (
                f"unexpected adapter param: {name}"
            )
    print(
        "  adapter: residual-path params (alpha + finger_logits) have non-zero "
        "grad; attention-path params have zero grad at alpha=0 init (expected) OK"
    )

    # 3. Aux head params (post + pre + pose) -- non-zero grad.
    for head_name, head in (
        ("aux_flow_post", model.aux_flow_post),
        ("aux_flow_pre", model.aux_flow_pre),
        ("aux_pose", model.aux_pose),
    ):
        if head is None:
            continue
        for n, p in head.named_parameters():
            assert p.grad is not None, f"{head_name}.{n} has grad=None"
            s = p.grad.abs().sum().item()
            assert s > 0.0, f"{head_name}.{n} grad sum = {s}"
        print(f"  {head_name}: all param grads non-None and non-zero OK")

    # 4. ModalityEmbedding bias.
    assert model.modality_embed.bias.grad is not None
    assert model.modality_embed.bias.grad.abs().sum().item() > 0.0
    print("  modality_embed.bias grad non-zero OK")

    # 5. VAE params must have grad is None (frozen, not just zero -- the
    #    distinction matters: requires_grad=False yields grad=None whereas
    #    requires_grad=True with zero loss yields grad=0).
    n_none = 0
    n_total = 0
    for n, p in model.vae.named_parameters():
        n_total += 1
        if p.grad is None:
            n_none += 1
        else:
            raise AssertionError(
                f"FROZEN VAE PARAM RECEIVED GRAD: vae.{n} "
                f"(sum={p.grad.abs().sum().item()})"
            )
    assert n_none == n_total
    print(f"  vae: ALL {n_total} params have grad is None (frozen) OK")

    # ------------------------------------------------------------------
    # Phase B: alpha != 0 -> attention path receives grad.
    #          Verifies the attention is wired correctly; the AdaLN-Zero
    #          init merely silences it at step 0, not forever.
    # ------------------------------------------------------------------
    _section("11b] Gradient flow (alpha=1.0): attention path opens up")
    _zero_grads(model)
    with torch.no_grad():
        model.adapter.alpha.fill_(1.0)
    out2 = model(tactile)
    loss2 = (
        F.mse_loss(out2["flow_pred_post"], flow_target_post)
        + 0.3 * F.mse_loss(out2["flow_pred_pre"], flow_target_pre)
        + 0.1 * F.mse_loss(out2["pose_pred"], pose_target)
    )
    loss2.backward()
    print(f"  loss (alpha=1.0): {float(loss2):.6f}")

    for name, p in model.adapter.named_parameters():
        assert p.grad is not None, f"adapter.{name} has grad=None at alpha=1"
        s = p.grad.abs().sum().item()
        assert s > 0.0, (
            f"adapter.{name} grad sum = {s} at alpha=1 -- attention path is "
            f"wired wrong (expected non-zero grad once alpha != 0)"
        )
    print("  adapter: ALL param grads non-zero once alpha != 0 OK")


# ---------------------------------------------------------------------------
# 12. latent_mode='mean' deterministic / 'sample' nondeterministic
# ---------------------------------------------------------------------------


def test_latent_mode_deterministic_vs_sample():
    _section("12] latent_mode: mean is deterministic, sample is not")
    torch.manual_seed(0)
    model = VisualVAEAdapterModel(
        vae=_make_random_vae(),
        spatial_h=2, spatial_w=2,
    )
    model.eval()
    tactile = torch.randn(1, 5, 1, 1, 64, 64).clamp(-1, 1)

    # Deterministic mode: two calls must give identical latents.
    with torch.no_grad():
        z1 = model.encode_per_finger(tactile, latent_mode="mean")
        z2 = model.encode_per_finger(tactile, latent_mode="mean")
    diff = (z1 - z2).abs().max().item()
    assert diff < 1e-6, f"latent_mode=mean is non-deterministic; max diff = {diff}"
    print(f"  mean: identical across calls (max diff {diff:.3e}) OK")

    # Sample mode: two calls almost surely differ.
    with torch.no_grad():
        zs1 = model.encode_per_finger(tactile, latent_mode="sample")
        zs2 = model.encode_per_finger(tactile, latent_mode="sample")
    diff_s = (zs1 - zs2).abs().max().item()
    assert diff_s > 1e-4, (
        f"latent_mode=sample should differ between calls; got {diff_s:.3e}"
    )
    print(f"  sample: differs across calls (max diff {diff_s:.3e}) OK")


# ---------------------------------------------------------------------------
# 12c. Stage-2 encode_per_hand API (F1a)
# ---------------------------------------------------------------------------


def test_encode_per_hand_shape():
    """F1a-1: shape contract for the multi-hand encode API.

    Input  ``(B, V_hand, F, T, H, W)`` 6-D grayscale tactile (no explicit C=1).
    Output ``(B, V_hand, C_lat, T_lat, H_lat, W_lat)`` per-hand fused latent.

    For the C5 contract at small spatial size (T=1, H=W=64 -> latent (2, 2)):
    output should be ``(B, V_hand, 128, 1, 2, 2)``.
    """
    _section("12c.i] encode_per_hand shape contract: (2, 2, 5, 1, 64, 64)")
    torch.manual_seed(0)
    model = VisualVAEAdapterModel(
        vae=_make_random_vae(),
        latent_channels=128,
        adapter_kind="finger_attention",
        spatial_h=2, spatial_w=2,
        enable_pre_fuse=False,
    )
    model.eval()

    # Multi-hand tactile clip: B=2 batch, V_hand=2 (left + right), F=5 fingers.
    tac = torch.randn(2, 2, 5, 1, 64, 64).clamp(-1, 1)
    z = model.encode_per_hand(tac)
    expected = (2, 2, 128, 1, 2, 2)
    assert z.shape == expected, f"shape {tuple(z.shape)} != {expected}"
    print(f"  out.shape = {tuple(z.shape)} OK")

    # V_hand=1 should also work (degenerate single-hand case).
    tac_1 = torch.randn(1, 1, 5, 1, 64, 64).clamp(-1, 1)
    z_1 = model.encode_per_hand(tac_1)
    assert z_1.shape == (1, 1, 128, 1, 2, 2), z_1.shape
    print(f"  V_hand=1: out.shape = {tuple(z_1.shape)} OK")


def test_encode_per_hand_matches_primitives():
    """F1a-2: bit-for-bit equivalence with encode_per_finger + fuse_per_hand.

    encode_per_hand is just a multi-hand wrapper; for V_hand=1, its output
    must equal ``fuse_per_hand(encode_per_finger(...))`` exactly under
    fixed-seed init. Catches any silent reshape / latent_mode drift in the
    new API. We use ``rtol=0, atol=0`` (i.e. ``torch.equal``) because the
    underlying ops are deterministic and identity-equivalent.
    """
    _section("12c.ii] encode_per_hand matches primitives bit-for-bit (V_hand=1)")
    torch.manual_seed(123)
    model = VisualVAEAdapterModel(
        vae=_make_random_vae(),
        latent_channels=128,
        adapter_kind="finger_attention",
        spatial_h=2, spatial_w=2,
        enable_pre_fuse=False,
        latent_mode="mean",  # deterministic
    )
    model.eval()

    tac = torch.randn(1, 5, 1, 1, 64, 64).clamp(-1, 1)             # 6-D, V_hand=1 spec'd later
    tac_per_hand = tac.unsqueeze(1).squeeze(3)                     # (1, 1, 5, 1, 64, 64)

    with torch.no_grad():
        # New API path.
        z_new = model.encode_per_hand(tac_per_hand)                # (1, 1, 128, 1, 2, 2)
        # Old primitive path (preserves Stage-1 reproducibility).
        z_per_finger = model.encode_per_finger(tac)                # (1, 5, 128, 1, 2, 2)
        z_old = model.fuse_per_hand(z_per_finger)                  # (1, 128, 1, 2, 2)

    diff = (z_new[:, 0] - z_old).abs().max().item()
    print(f"  max|encode_per_hand[V=0] - fuse_per_hand(...)| = {diff:.3e}")
    assert diff == 0.0, (
        f"bit-for-bit equivalence broken (diff={diff:.3e}). The new API has "
        f"behavioral drift; existing Stage-1 forward() reproducibility is at "
        f"risk."
    )
    print("  PASS: V_hand=1 slice matches primitives EXACTLY")


def test_encode_per_hand_no_grad_invariant():
    """F1a-3: @torch.no_grad() decorator means returned tensor cannot be
    backpropagated.

    Stage 2 freezes the entire VisualVAEAdapterModel. The decorator is a
    defense-in-depth guard against an accidental gradient path; a Stage-2
    trainer that wants to fine-tune v0c-A in the future MUST switch to the
    primitive ``encode_per_finger + fuse_per_hand`` path explicitly.
    """
    _section("12c.iii] encode_per_hand: requires_grad invariant")
    torch.manual_seed(0)
    model = VisualVAEAdapterModel(
        vae=_make_random_vae(),
        latent_channels=128,
        adapter_kind="finger_attention",
        spatial_h=2, spatial_w=2,
        enable_pre_fuse=False,
    )
    # Note: model.train() vs model.eval() doesn't matter for autograd; the
    # @torch.no_grad() decorator is what matters.
    model.train()

    tac = torch.randn(1, 1, 5, 1, 64, 64).clamp(-1, 1)
    tac.requires_grad_(False)
    z = model.encode_per_hand(tac)
    assert z.requires_grad is False, (
        f"encode_per_hand output should have requires_grad=False (no_grad "
        f"decorator); got requires_grad={z.requires_grad}"
    )
    # Calling .sum().backward() must raise (no graph).
    try:
        z.sum().backward()
    except RuntimeError as e:
        print(f"  backward raises RuntimeError as expected: {type(e).__name__} OK")
    else:
        raise AssertionError(
            "expected RuntimeError on backward through @torch.no_grad output"
        )


def test_encode_per_hand_input_validation():
    """F1a-4: input shape / num_fingers validation raises clear ValueErrors.

    Wiring bugs in the trainer / dataset are easier to debug if the API
    fails fast on shape mismatches rather than silently broadcasting.
    """
    _section("12c.iv] encode_per_hand input validation")
    model = VisualVAEAdapterModel(
        vae=_make_random_vae(),
        latent_channels=128,
        adapter_kind="finger_attention",
        spatial_h=2, spatial_w=2,
        enable_pre_fuse=False,
    )
    model.eval()
    # Wrong ndim.
    try:
        model.encode_per_hand(torch.randn(1, 5, 1, 64, 64))    # 5-D
    except ValueError as e:
        print(f"  5-D input -> ValueError OK: {str(e)[:60]}...")
    else:
        raise AssertionError("expected ValueError for 5-D input")

    # Wrong F.
    try:
        model.encode_per_hand(torch.randn(1, 1, 7, 1, 64, 64))  # F=7 != 5
    except ValueError as e:
        print(f"  F=7 input -> ValueError OK: {str(e)[:60]}...")
    else:
        raise AssertionError("expected ValueError for F=7")


# ---------------------------------------------------------------------------
# 13. YAML dry-run: parse v0 yaml + construct the wrapper from its config
# ---------------------------------------------------------------------------


def test_yaml_config_parse_and_construct():
    """Parse the v0 yaml and instantiate VisualVAEAdapterModel from it.

    This is the structural counterpart to :func:`test_wrapper_forward_small`
    that confirms the YAML the trainer will load is shape-compatible with
    the wrapper. Catches schema drift (renamed keys, removed fields) before
    a full GPU training launch surfaces it as a runtime error.
    """
    _section("13] YAML dry-run: parse v0 yaml + construct wrapper")

    yaml_path = os.path.join(
        REPO_ROOT, "configs", "tactile_model",
        "stage1_lite_visual_vae_adapter_v0.yaml",
    )
    assert os.path.isfile(yaml_path), f"yaml missing: {yaml_path}"

    from yaml import Loader, load
    with open(yaml_path) as f:
        cfg = load(f, Loader=Loader)

    # Top-level required keys (used by the trainer / launcher).
    required_top = [
        "output_dir", "visual_vae_path", "visual_vae_path_kind",
        "mixed_precision", "lr", "batch_size", "train_steps",
        "steps_to_log", "steps_to_val", "steps_to_save",
        "best_ckpt_metric", "tactile_vae", "data",
    ]
    for k in required_top:
        assert k in cfg, f"yaml missing required top-level key: {k}"
    assert cfg["best_ckpt_metric"] == "val_flow_mse_active_mean", (
        f"v0 yaml must select on val_flow_mse_active_mean; got "
        f"{cfg['best_ckpt_metric']!r}"
    )
    assert cfg["visual_vae_path_kind"] in {"subfolder", "direct"}, (
        f"yaml: visual_vae_path_kind={cfg['visual_vae_path_kind']!r}"
    )

    adapter_cfg = cfg["tactile_vae"]["config"]
    required_adapter = [
        "adapter_kind", "latent_channels", "num_fingers", "spatial_h",
        "spatial_w", "adapter_n_heads", "adapter_use_finger_embed",
        "adapter_use_pos_query", "adapter_residual", "gray_to_rgb_init",
        "vae_latent_mode", "enable_pre_fuse",
        "lambda_loc", "lambda_loc_pre", "lambda_pose", "lambda_kl",
        "flow_loss",
    ]
    for k in required_adapter:
        assert k in adapter_cfg, f"yaml: tactile_vae.config missing key: {k}"
    assert adapter_cfg["latent_channels"] == 128, (
        f"latent_channels must be 128 for the LTX C5 contract; got "
        f"{adapter_cfg['latent_channels']}"
    )
    assert adapter_cfg["spatial_h"] == 6 and adapter_cfg["spatial_w"] == 8, (
        f"spatial_h/w must be (6, 8) for the LTX C5 contract; got "
        f"({adapter_cfg['spatial_h']}, {adapter_cfg['spatial_w']})"
    )
    assert float(adapter_cfg["lambda_kl"]) == 0.0, (
        f"lambda_kl must be 0 for the lite trainer (frozen VAE); got "
        f"{adapter_cfg['lambda_kl']}"
    )

    flow_cfg = adapter_cfg["flow_loss"]
    assert flow_cfg.get("enabled") is True
    assert flow_cfg.get("form") == "sqrt_magnitude_aware"
    print("  schema OK")

    # Build wrapper from the yaml-derived kwargs (mirrors
    # VisualVAEAdapterTrainer.prepare_models exactly).
    wrapper_kwargs = {
        "vae":              _make_random_vae(),
        "latent_channels":  int(adapter_cfg["latent_channels"]),
        "adapter_kind":     str(adapter_cfg["adapter_kind"]),
        "num_fingers":      int(adapter_cfg["num_fingers"]),
        "num_heads":        int(adapter_cfg["adapter_n_heads"]),
        "spatial_h":        int(adapter_cfg["spatial_h"]),
        "spatial_w":        int(adapter_cfg["spatial_w"]),
        "use_finger_embed": bool(adapter_cfg["adapter_use_finger_embed"]),
        "use_pos_query":    bool(adapter_cfg["adapter_use_pos_query"]),
        "adapter_residual": str(adapter_cfg["adapter_residual"]),
        "gray_to_rgb_init": str(adapter_cfg["gray_to_rgb_init"]),
        "latent_mode":      str(adapter_cfg["vae_latent_mode"]),
        "enable_pre_fuse":  bool(adapter_cfg["enable_pre_fuse"]),
    }
    model = VisualVAEAdapterModel(**wrapper_kwargs)

    # Frozen-VAE invariant.
    n_vae_train = sum(p.numel() for p in model.vae.parameters() if p.requires_grad)
    assert n_vae_train == 0, (
        f"frozen-VAE invariant violated: {n_vae_train} VAE params with "
        f"requires_grad=True after construction from yaml"
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  yaml-built wrapper trainable params: {n_train/1e6:.2f} M  OK")

    # NOTE: we deliberately do NOT run a forward pass here. The YAML pins
    # `spatial_h=6, spatial_w=8` (the LTX C5 contract) which requires
    # full-res 192x256 tactile input -- that's the slow path covered by
    # `test_wrapper_c5_contract_full_res` (gated behind `--quick`). The
    # smaller-resolution forward sanity is covered by `test_wrapper_forward_small`
    # which intentionally bypasses the YAML's spatial config.
    expected_h = int(adapter_cfg["spatial_h"])
    expected_w = int(adapter_cfg["spatial_w"])
    assert (model.adapter.h, model.adapter.w) == (expected_h, expected_w), (
        f"adapter spatial config mismatch: yaml=({expected_h}, {expected_w}) "
        f"vs adapter=({model.adapter.h}, {model.adapter.w})"
    )
    print(f"  adapter pinned to ({expected_h}, {expected_w}) per yaml OK")


# ---------------------------------------------------------------------------
# 14. Trainer module imports cleanly (no circular imports / typo crashes)
# ---------------------------------------------------------------------------


def test_trainer_module_imports():
    """Import :mod:`runner.visual_vae_adapter_trainer` and assert the public
    symbol is reachable. Does NOT instantiate the trainer (which would need
    an accelerator + dataset).
    """
    _section("14] Trainer module imports cleanly")
    try:
        import runner.visual_vae_adapter_trainer as mod
    except Exception as e:
        raise AssertionError(
            f"importing runner.visual_vae_adapter_trainer raised "
            f"{type(e).__name__}: {e}"
        )
    assert hasattr(mod, "VisualVAEAdapterTrainer"), (
        "VisualVAEAdapterTrainer not exported from "
        "runner.visual_vae_adapter_trainer"
    )
    assert hasattr(mod, "_V4BAccumulator"), (
        "_V4BAccumulator helper not exported"
    )
    print(f"  imported {mod.__name__} OK; "
          f"VisualVAEAdapterTrainer={mod.VisualVAEAdapterTrainer.__name__}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip the slow full-res C5 contract test.",
    )
    args = parser.parse_args()

    print(f"torch={torch.__version__}")

    test_gray_to_rgb_init_ones_repeat()
    test_gray_to_rgb_init_uniform_third()

    test_finger_attention_shape()
    test_finger_attention_alpha_zero_starts_at_residual()
    test_finger_attention_finger_logits_zero_equals_mean_pool()
    test_finger_attention_finger_embed_used()
    test_finger_attention_pos_query_used()

    test_concat_channel_adapter_shape()

    test_finger_set_transformer_shape()
    test_finger_set_transformer_param_count_dominates_v1()
    test_finger_set_transformer_alpha_zero_starts_at_residual()
    test_finger_set_transformer_gradient_flow()
    test_wrapper_finger_set_transformer_e2e()

    # v0d: pose injection + TimeSformer + dropped AuxPoseDecoder (section 6c).
    test_v0d_param_count()
    test_v0d_alpha_pose_zero_equals_v0c_a()
    test_v0d_alpha_temp_zero_equals_v0c_a()
    test_v0d_both_alphas_zero_equals_v0c_a()
    test_v0d_gradient_flow_after_alpha_bumps()
    test_v0d_wrapper_drops_aux_pose()
    test_v0d_wrapper_rejects_v0d_kwargs_on_other_adapters()
    test_v0d_align_pose_to_lat()

    test_aux_pre_fuse_head_shape()
    test_aux_flow_decoder_shape()
    test_aux_pose_decoder_shape()

    test_wrapper_frozen_vae_and_param_breakdown()
    test_wrapper_forward_small()

    if not args.quick:
        test_wrapper_c5_contract_full_res()
    else:
        print("\n[10] SKIPPED (--quick): full-res C5 contract")

    test_wrapper_gradient_flow()
    test_latent_mode_deterministic_vs_sample()

    # F1a (Stage 2 encode_per_hand API).
    test_encode_per_hand_shape()
    test_encode_per_hand_matches_primitives()
    test_encode_per_hand_no_grad_invariant()
    test_encode_per_hand_input_validation()

    test_yaml_config_parse_and_construct()
    test_trainer_module_imports()

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
