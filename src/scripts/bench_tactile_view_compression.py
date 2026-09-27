#!/usr/bin/env python3
"""Benchmark the system cost of tactile view compression in DexVTAM.

Motivation
----------
DexVTAM compresses the 10 per-finger tactile latents (5 fingers x 2 hands)
into 2 per-hand latent views with the ``FingerSetTransformerAdapter`` before
the shared LTX DiT / action expert. The DiT's cost is dominated by the number
of *view tokens* it attends over:

    tokens = n_view * (T_lat * H_lat * W_lat)

so the compressed deployment path runs with

    V = 3   (1 head visual view + 2 per-hand tactile views)

while an *uncompressed* per-finger formulation would need

    V = 11  (1 head visual view + 10 per-finger tactile views).

This script measures peak GPU memory and per-step latency of the *exact*
deployment forward path (``LTXVideoTransformer3DModel.forward`` with the same
kwargs ``custom_pipeline.infer`` uses) while sweeping ``n_view``. Because the
cost is entirely shape-driven, random weights + synthetic latents are a valid
and standard way to isolate the systems cost -- this is NOT an accuracy
benchmark. See ``docs/tactile_view_compression_benchmark.md``.

Two things are measured:
  1. DiT / action-expert cost as a function of ``n_view`` (main result).
  2. (optional) FingerSetTransformerAdapter fusion overhead -- the extra cost
     paid ONLY by the compressed V=3 path to turn 10 finger latents into 2
     hand latents. Reported in isolation so it does not pollute the DiT peak.

The forward loop mirrors deployment: step 0 runs the full world-model forward
(``return_video=True, store_buffer=True``) and caches per-block video states;
steps 1..N-1 reuse the cached buffer (``return_video=False``) and only run the
action expert -- exactly like ``custom_pipeline.infer``.
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time

import numpy as np
import yaml

# Allow `from models...` / `from utils...` when run as `python scripts/xxx.py`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from models.ltx_models.transformer_ltx_multiview import (  # noqa: E402
    LTXVideoTransformer3DModel,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def reset_peak():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def bytes_to_gb(x):
    return round(x / 1024**3, 4)


# --------------------------------------------------------------------------- #
# Model build (random weights, max_view overridden to fit the largest V)
# --------------------------------------------------------------------------- #
def build_transformer(cfg, max_view, dtype, device):
    model_kwargs = dict(cfg["diffusion_model"]["config"])
    # Override max_view so the view-embed table is large enough for V=11.
    model_kwargs["max_view"] = int(max_view)
    model = LTXVideoTransformer3DModel(**model_kwargs)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, model_kwargs


def build_adapter(cfg, dtype, device):
    from models.tactile_models.visual_vae_adapter import FingerSetTransformerAdapter

    tac = cfg["tactile_vae"]["config"]
    adapter = FingerSetTransformerAdapter(
        embed_dim=tac["latent_channels"],
        num_fingers=tac["num_fingers"],
        num_heads=tac["num_heads"],
        num_layers=tac["adapter_n_layers"],
        ffn_dim=tac["adapter_ffn_dim"],
        dropout=tac.get("adapter_dropout", 0.0),
        h=tac["spatial_h"],
        w=tac["spatial_w"],
        use_finger_embed=tac.get("use_finger_embed", True),
        use_pos_query=tac.get("use_pos_query", True),
        residual=tac.get("adapter_residual", "weighted_mean"),
        use_pose_injection=tac.get("adapter_use_pose_injection", False),
        pose_dim=tac.get("adapter_pose_dim", 22),
        use_timesformer=tac.get("adapter_use_timesformer", False),
        timesformer_num_blocks=tac.get("adapter_timesformer_num_blocks", 2),
        timesformer_num_heads=tac.get("adapter_timesformer_num_heads", 8),
        timesformer_ffn_dim=tac.get("adapter_timesformer_ffn_dim", 512),
        finger_dropout=tac.get("adapter_finger_dropout", 0.0),
    )
    adapter = adapter.to(device=device, dtype=dtype)
    adapter.eval()
    return adapter, tac


def build_vae(vae_dir, dtype, device):
    """Build the real LTX VAE (used only in --e2e mode).

    Weights don't affect latency (shape-driven), but from_pretrained is the
    faithful way to get the exact encoder architecture. Point --vae_dir at a
    local ltx_video dir or an HF id (default: Lightricks/LTX-Video).
    """
    from models.ltx_models.autoencoder_kl_ltx import AutoencoderKLLTXVideo

    vae = AutoencoderKLLTXVideo.from_pretrained(
        vae_dir, subfolder="vae", torch_dtype=dtype
    )
    vae = vae.to(device=device, dtype=dtype)
    vae.eval()
    return vae


def _time_ms(fn, device):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e)


# --------------------------------------------------------------------------- #
# One deploy-style N-step action denoise, timed (first step vs cached steps)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_denoise_once(model, dims, num_steps, device, dtype, gen):
    B = dims["batch_size"]
    V = dims["n_view"]
    n_visual = dims["n_view_visual"]
    L = dims["tokens_per_view"]
    C_lat = dims["in_channels"]
    C_text = dims["caption_channels"]
    L_text = dims["l_text"]
    action_chunk = dims["action_chunk"]
    action_dim = dims["action_out_channels"]
    c_hist = dims["action_in_channels"]
    add_state = dims["add_state"]
    num_frames = dims["t_lat"]
    height = dims["h_lat"]
    width = dims["w_lat"]
    rope_scale = dims["rope_interpolation_scale"]

    # Synthetic latents / conditioning (values irrelevant; shapes are the point)
    hidden = torch.randn(B * V, L, C_lat, device=device, dtype=dtype, generator=gen)
    enc = torch.randn(B, L_text, C_text, device=device, dtype=dtype, generator=gen)
    actions = torch.randn(
        B, action_chunk, action_dim, device=device, dtype=dtype, generator=gen
    )
    history_action_state = None
    if add_state:
        history_action_state = torch.randn(
            B, 1, c_hist, device=device, dtype=dtype, generator=gen
        )

    common = dict(
        hidden_states=hidden,
        encoder_hidden_states=enc,
        encoder_attention_mask=None,
        num_frames=num_frames,
        height=height,
        width=width,
        rope_interpolation_scale=rope_scale,
        return_dict=False,
        return_action=True,
        n_view=V,
        n_view_visual=n_visual,
        video_attention_mask=None,
    )

    ev0s, ev0e = torch.cuda.Event(True), torch.cuda.Event(True)
    ev1s, ev1e = torch.cuda.Event(True), torch.cuda.Event(True)

    def _t(i):
        val = 1000.0 * (1.0 - i / num_steps)
        ts_video = torch.full((B * V, L), val, device=device, dtype=dtype)
        ts_action = torch.full((B, action_chunk), val, device=device, dtype=dtype)
        return ts_video, ts_action

    # ---- step 0: full WM forward + store buffer + action ----
    ts_video, ts_action = _t(0)
    ev0s.record()
    out = model(
        timestep=ts_video,
        action_states=actions,
        action_timestep=ts_action,
        return_video=True,
        store_buffer=True,
        video_states_buffer=None,
        history_action_state=history_action_state,
        **common,
    )[0]
    video_states_buffer = out["video_states_buffer"]
    actions = actions + 0.01 * out["action"].to(dtype)
    ev0e.record()

    # ---- steps 1..N-1: action expert only, reuse cached video states ----
    ev1s.record()
    for i in range(1, num_steps):
        ts_video, ts_action = _t(i)
        out = model(
            timestep=ts_video,
            action_states=actions,
            action_timestep=ts_action,
            return_video=False,
            store_buffer=False,
            video_states_buffer=video_states_buffer,
            history_action_state=history_action_state,
            **common,
        )[0]
        actions = actions + 0.01 * out["action"].to(dtype)
    ev1e.record()

    torch.cuda.synchronize()
    first_ms = ev0s.elapsed_time(ev0e)
    cached_ms = ev1s.elapsed_time(ev1e)
    return first_ms, cached_ms, first_ms + cached_ms


@torch.no_grad()
def benchmark_decompose(model, dims, args, device, dtype):
    """Split the denoise loop into WM-forward vs action-head so they sum to the
    measured total:  total ~= wm_forward + N * action_step.

    - wm_forward : step-0 world-model DiT forward (return_action=False,
      store_buffer=True) — the one-time cost that caches per-block video states.
    - action_step: one action-expert step over the cached video states
      (return_video=False) — this runs N times in deployment.
    """
    B = dims["batch_size"]
    V = dims["n_view"]
    n_visual = dims["n_view_visual"]
    L = dims["tokens_per_view"]
    C_lat = dims["in_channels"]
    C_text = dims["caption_channels"]
    L_text = dims["l_text"]
    action_chunk = dims["action_chunk"]
    action_dim = dims["action_out_channels"]
    c_hist = dims["action_in_channels"]
    add_state = dims["add_state"]
    num_frames = dims["t_lat"]
    height = dims["h_lat"]
    width = dims["w_lat"]
    rope_scale = dims["rope_interpolation_scale"]
    N = args.num_inference_steps
    gen = torch.Generator(device=device).manual_seed(args.seed)

    hidden = torch.randn(B * V, L, C_lat, device=device, dtype=dtype, generator=gen)
    enc = torch.randn(B, L_text, C_text, device=device, dtype=dtype, generator=gen)
    actions = torch.randn(B, action_chunk, action_dim, device=device, dtype=dtype, generator=gen)
    hist = None
    if add_state:
        hist = torch.randn(B, 1, c_hist, device=device, dtype=dtype, generator=gen)
    ts_v = torch.full((B * V, L), 500.0, device=device, dtype=dtype)
    ts_a = torch.full((B, action_chunk), 500.0, device=device, dtype=dtype)
    common = dict(
        hidden_states=hidden, encoder_hidden_states=enc, encoder_attention_mask=None,
        num_frames=num_frames, height=height, width=width,
        rope_interpolation_scale=rope_scale, return_dict=False,
        n_view=V, n_view_visual=n_visual, video_attention_mask=None,
    )

    def _wm_only():
        return model(timestep=ts_v, action_states=None, action_timestep=None,
                     return_video=True, return_action=False, store_buffer=True,
                     video_states_buffer=None, history_action_state=None, **common)[0]

    def _action_step(buf):
        return model(timestep=ts_v, action_states=actions, action_timestep=ts_a,
                     return_video=False, return_action=True, store_buffer=False,
                     video_states_buffer=buf, history_action_state=hist, **common)[0]

    # warmup (full realistic pattern)
    for _ in range(args.warmup):
        buf = _wm_only()["video_states_buffer"]
        _action_step(buf)

    wm_ms, act_ms = [], []
    for _ in range(args.iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        buf = _wm_only()["video_states_buffer"]
        e.record()
        torch.cuda.synchronize()
        wm_ms.append(s.elapsed_time(e))

        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        _action_step(buf)
        e.record()
        torch.cuda.synchronize()
        act_ms.append(s.elapsed_time(e))

    wm = statistics.mean(wm_ms)
    act = statistics.mean(act_ms)
    return {
        "n_view": V,
        "dit_tokens": V * L,
        "wm_forward_ms": round(wm, 3),
        "action_step_ms": round(act, 3),
        "num_steps": N,
        "reconstructed_total_ms": round(wm + N * act, 3),
        "action_share_pct": round(100.0 * (N * act) / (wm + N * act), 1),
    }


def benchmark_view(model, dims, args, device, dtype):
    gen = torch.Generator(device=device).manual_seed(args.seed)
    # Warmup
    for _ in range(args.warmup):
        run_denoise_once(model, dims, args.num_inference_steps, device, dtype, gen)

    reset_peak()

    firsts, cacheds, totals = [], [], []
    for _ in range(args.iters):
        f, c, t = run_denoise_once(
            model, dims, args.num_inference_steps, device, dtype, gen
        )
        firsts.append(f)
        cacheds.append(c)
        totals.append(t)

    torch.cuda.synchronize()
    peak_alloc = bytes_to_gb(torch.cuda.max_memory_allocated(device))
    peak_reserved = bytes_to_gb(torch.cuda.max_memory_reserved(device))

    def _ms(x):
        return round(statistics.mean(x), 3)

    def _std(x):
        return round(statistics.pstdev(x), 3) if len(x) > 1 else 0.0

    return {
        "n_view": dims["n_view"],
        "n_view_visual": dims["n_view_visual"],
        "tokens_per_view": dims["tokens_per_view"],
        "dit_tokens": dims["n_view"] * dims["tokens_per_view"],
        "peak_alloc_gb": peak_alloc,
        "peak_reserved_gb": peak_reserved,
        "first_step_ms": _ms(firsts),
        "first_step_ms_std": _std(firsts),
        "cached_steps_ms": _ms(cacheds),
        "cached_steps_ms_std": _std(cacheds),
        "total_ms": _ms(totals),
        "total_ms_std": _std(totals),
    }


# --------------------------------------------------------------------------- #
# Adapter fusion overhead (compressed-path-only cost), measured in isolation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def benchmark_adapter(cfg, args, device, dtype):
    adapter, tac = build_adapter(cfg, dtype, device)
    n_hands = cfg.get("projector", {}).get("num_views", 2)
    F = tac["num_fingers"]
    C = tac["latent_channels"]
    H = tac["spatial_h"]
    W = tac["spatial_w"]
    T = args.t_lat
    pose_dim = tac.get("adapter_pose_dim", 22)
    use_pose = tac.get("adapter_use_pose_injection", False)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    B = args.batch_size * n_hands  # one set of tokens per hand

    def _once():
        z = torch.randn(B, F, C, T, H, W, device=device, dtype=dtype, generator=gen)
        pose = None
        if use_pose:
            pose = torch.randn(B, T, pose_dim, device=device, dtype=dtype, generator=gen)
        return adapter(z, hand_pose=pose)

    for _ in range(args.warmup):
        _once()

    reset_peak()
    ev_s, ev_e = torch.cuda.Event(True), torch.cuda.Event(True)
    times = []
    for _ in range(args.iters):
        ev_s.record()
        _once()
        ev_e.record()
        torch.cuda.synchronize()
        times.append(ev_s.elapsed_time(ev_e))
    peak_alloc = bytes_to_gb(torch.cuda.max_memory_allocated(device))

    result = {
        "n_hands": n_hands,
        "num_fingers": F,
        "latent_grid": [T, H, W],
        "fuse_ms": round(statistics.mean(times), 3),
        "fuse_ms_std": round(statistics.pstdev(times), 3) if len(times) > 1 else 0.0,
        "peak_alloc_gb": peak_alloc,
    }
    del adapter
    reset_peak()
    return result


# --------------------------------------------------------------------------- #
# Full per-chunk e2e: VAE-encode(obs) + tactile fuse + denoise(WM + action)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def benchmark_e2e(model, vae, adapter, cfg, dims_base, args, device, dtype, compressed):
    """One deployment chunk end-to-end (excluding cached T5 text encode):

      stage 1  VAE encode observations (head mem frames + all finger mem frames)
      stage 2  tactile adapter fusion (10 finger latents -> 2 hand latents)   [compressed only]
      stage 3  denoise loop = WM step-0 + action expert x N   (INCLUDES action model)

    compressed:   V = 1 head + n_hands tactile views (=3), pays stage 2
    uncompressed: V = 1 head + (n_hands*n_fingers) finger views (=11), skips stage 2
    Stage 1 (VAE) is identical for both -- all finger images must be encoded
    either way; it is the shared, view-independent fixed cost.
    """
    tac = cfg["tactile_vae"]["config"]
    n_hands = cfg.get("projector", {}).get("num_views", 2)
    n_fingers = tac["num_fingers"]
    mem = int(cfg["data"]["train"]["n_previous"])
    sh, sw = cfg["data"]["train"]["sample_size"]
    n_visual = dims_base["n_view_visual"]
    C_lat = dims_base["in_channels"]
    h_lat, w_lat = dims_base["h_lat"], dims_base["w_lat"]
    pose_dim = tac.get("adapter_pose_dim", 22)
    use_pose = tac.get("adapter_use_pose_injection", False)

    V = n_visual + (n_hands if compressed else n_hands * n_fingers)
    dims = dict(dims_base)
    dims["n_view"] = V

    # All observation frames encoded per-frame at native resolution. Head =
    # n_visual*mem frames; tactile = n_hands*n_fingers*mem finger frames.
    n_obs_frames = (n_visual + n_hands * n_fingers) * mem
    gen = torch.Generator(device=device).manual_seed(args.seed)

    def _vae_encode():
        x = torch.randn(n_obs_frames, 3, 1, sh, sw, device=device, dtype=dtype, generator=gen)
        return vae.encode(x).latent_dist.mode()

    def _fuse():
        z = torch.randn(
            n_hands, n_fingers, C_lat, mem, h_lat, w_lat,
            device=device, dtype=dtype, generator=gen,
        )
        pose = None
        if use_pose:
            pose = torch.randn(n_hands, mem, pose_dim, device=device, dtype=dtype, generator=gen)
        return adapter(z, hand_pose=pose)

    def _one_chunk():
        _vae_encode()
        if compressed:
            _fuse()
        run_denoise_once(model, dims, args.num_inference_steps, device, dtype, gen)

    for _ in range(args.warmup):
        _one_chunk()

    reset_peak()
    vae_ms, fuse_ms, denoise_ms, total_ms = [], [], [], []
    for _ in range(args.iters):
        vae_ms.append(_time_ms(_vae_encode, device))
        if compressed:
            fuse_ms.append(_time_ms(_fuse, device))
        f, c, t = run_denoise_once(model, dims, args.num_inference_steps, device, dtype, gen)
        denoise_ms.append(t)
        total_ms.append((vae_ms[-1]) + (fuse_ms[-1] if compressed else 0.0) + t)
    peak_alloc = bytes_to_gb(torch.cuda.max_memory_allocated(device))
    reset_peak()

    def _m(x):
        return round(statistics.mean(x), 3) if x else 0.0

    return {
        "variant": "compressed" if compressed else "uncompressed",
        "n_view": V,
        "n_view_visual": n_visual,
        "dit_tokens": V * dims_base["tokens_per_view"],
        "obs_frames_encoded": n_obs_frames,
        "vae_encode_ms": _m(vae_ms),
        "adapter_fuse_ms": _m(fuse_ms) if compressed else 0.0,
        "denoise_ms": _m(denoise_ms),
        "e2e_total_ms": _m(total_ms),
        "peak_alloc_gb": peak_alloc,
    }


def write_e2e_markdown(path, rows, meta):
    comp = next((r for r in rows if r["variant"] == "compressed"), None)
    unc = next((r for r in rows if r["variant"] == "uncompressed"), None)
    lines = ["# Tactile View Compression -- End-to-End Latency (results)\n"]
    lines.append(
        f"- GPU: `{meta['gpu']}`  |  dtype: `{meta['dtype']}`  |  batch: "
        f"{meta['batch_size']}  |  steps: {meta['num_inference_steps']}  |  "
        f"iters: {meta['iters']} (warmup {meta['warmup']})\n"
    )
    lines.append(f"- config: `{meta['config']}`  |  VAE: `{meta['vae_dir']}`\n")
    lines.append(
        "- e2e = VAE-encode(obs) + adapter fusion + denoise loop (WM step-0 + "
        "action expert x N). Excludes cached T5 text encode.\n"
    )
    lines.append("")
    lines.append(
        "| variant | V | DiT tokens | obs frames | VAE encode (ms) | "
        "adapter fuse (ms) | denoise WM+action (ms) | **e2e total (ms)** | peak alloc (GB) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['n_view']} | {r['dit_tokens']} | "
            f"{r['obs_frames_encoded']} | {r['vae_encode_ms']} | "
            f"{r['adapter_fuse_ms']} | {r['denoise_ms']} | "
            f"**{r['e2e_total_ms']}** | {r['peak_alloc_gb']} |"
        )
    lines.append("")
    if comp and unc and comp["e2e_total_ms"] > 0:
        lines.append(
            f"e2e speedup (uncompressed / compressed): "
            f"**{unc['e2e_total_ms'] / comp['e2e_total_ms']:.2f}x**  "
            f"({unc['e2e_total_ms']} -> {comp['e2e_total_ms']} ms); "
            f"denoise-only: {unc['denoise_ms'] / comp['denoise_ms']:.2f}x; "
            f"VAE encode is a shared {comp['vae_encode_ms']} ms fixed cost.\n"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f">>> wrote e2e markdown: {path}")


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_markdown(path, rows, adapter_row, meta):
    base = next((r for r in rows if r["n_view"] == 3), rows[0])
    base_tokens = base["dit_tokens"]
    lines = []
    lines.append("# Tactile View Compression Benchmark (results)\n")
    lines.append(
        f"- GPU: `{meta['gpu']}`  |  dtype: `{meta['dtype']}`  |  "
        f"batch: {meta['batch_size']}  |  steps: {meta['num_inference_steps']}  |  "
        f"iters: {meta['iters']} (warmup {meta['warmup']})\n"
    )
    lines.append(f"- config: `{meta['config']}`\n")
    lines.append(f"- weights: `{meta['weights']}`\n")
    lines.append("")
    lines.append(
        "| V | views | DiT tokens | token ratio vs V=3 | peak alloc (GB) | "
        "peak reserved (GB) | total 10-step (ms) | first step (ms) | "
        "cached steps (ms) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r.get("oom"):
            lines.append(
                f"| {r['n_view']} | {r['n_view_visual']} head + "
                f"{r['n_view'] - r['n_view_visual']} tactile | "
                f"{r['dit_tokens']} | {r['dit_tokens'] / base_tokens:.2f}x | "
                f"**OOM** | OOM | OOM | OOM | OOM |"
            )
            continue
        lines.append(
            f"| {r['n_view']} | {r['n_view_visual']} head + "
            f"{r['n_view'] - r['n_view_visual']} tactile | "
            f"{r['dit_tokens']} | {r['dit_tokens'] / base_tokens:.2f}x | "
            f"{r['peak_alloc_gb']} | {r['peak_reserved_gb']} | "
            f"{r['total_ms']} +/- {r['total_ms_std']} | "
            f"{r['first_step_ms']} | {r['cached_steps_ms']} |"
        )
    lines.append("")
    if adapter_row is not None:
        lines.append(
            f"**FingerSetTransformerAdapter fusion overhead** "
            f"(10 finger latents -> {adapter_row['n_hands']} hand latents): "
            f"{adapter_row['fuse_ms']} +/- {adapter_row['fuse_ms_std']} ms, "
            f"{adapter_row['peak_alloc_gb']} GB peak alloc. Paid only by the "
            f"compressed V=3 path.\n"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f">>> wrote markdown: {path}")


def maybe_plot(png_prefix, rows):
    valid = [r for r in rows if not r.get("oom")]
    if len(valid) < 2:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f">>> skip plots (matplotlib unavailable): {e}")
        return
    vs = [r["n_view"] for r in valid]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(vs, [r["peak_alloc_gb"] for r in valid], "o-")
    ax.set_xlabel("n_view (V)")
    ax.set_ylabel("peak allocated memory (GB)")
    ax.set_title("Peak memory vs view count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{png_prefix}_memory.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(vs, [r["total_ms"] for r in valid], "o-", label="total 10-step")
    ax.plot(vs, [r["first_step_ms"] for r in valid], "s--", label="first step")
    ax.plot(vs, [r["cached_steps_ms"] for r in valid], "^--", label="cached steps")
    ax.set_xlabel("n_view (V)")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Latency vs view count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{png_prefix}_latency.png", dpi=150)
    plt.close(fig)
    print(f">>> wrote plots: {png_prefix}_memory.png / {png_prefix}_latency.png")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default="configs/ltx_model/diverse_488/action_model_diverse_488_tactile_v0d.yaml",
        help="action-model YAML to read dims from (dims only; weights random).",
    )
    p.add_argument(
        "--n_views",
        type=int,
        nargs="+",
        default=[3, 5, 7, 9, 11],
        help="view counts to sweep. Compressed=3 (1 head+2 hands), "
        "uncompressed=11 (1 head+10 fingers).",
    )
    p.add_argument("--n_visual", type=int, default=1, help="# visual (head) views.")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--dtype", choices=list(DTYPE_MAP), default="bf16")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--t_lat", type=int, default=6, help="latent temporal frames.")
    p.add_argument("--l_text", type=int, default=128, help="# text tokens.")
    p.add_argument("--frame_rate", type=float, default=30.0)
    p.add_argument("--temporal_ratio", type=int, default=8)
    p.add_argument("--spatial_ratio", type=int, default=32)
    p.add_argument("--measure_adapter", action="store_true")
    p.add_argument(
        "--decompose",
        action="store_true",
        help="split denoise into WM-forward vs action-head per step "
        "(total ~= wm_forward + N * action_step) for each V.",
    )
    p.add_argument(
        "--e2e",
        action="store_true",
        help="measure full per-chunk e2e latency (VAE encode + adapter fusion "
        "+ denoise WM+action) for compressed V=3 vs uncompressed V=11.",
    )
    p.add_argument(
        "--vae_dir",
        default="Lightricks/LTX-Video",
        help="LTX VAE source for --e2e (local ltx_video dir or HF id). Weights "
        "don't affect latency; only the encoder architecture matters.",
    )
    p.add_argument(
        "--isolate_each_view",
        action="store_true",
        help="run exactly ONE V (requires --n_views to contain a single value) "
        "and exit; the shell loops over V in separate processes for "
        "trustworthy reserved-memory numbers.",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="optional DiT checkpoint for sanity checking. The reported "
        "systems benchmark uses random weights unless this is set.",
    )
    p.add_argument("--out_dir", default="docs")
    p.add_argument("--tag", default="", help="suffix for output filenames.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA required."
    device = torch.device("cuda")
    dtype = DTYPE_MAP[args.dtype]

    if args.isolate_each_view and len(args.n_views) != 1:
        raise ValueError(
            "--isolate_each_view requires exactly one value in --n_views; "
            f"got {args.n_views}. The shell launcher loops over V, calling "
            "this script once per V in a fresh process."
        )

    cfg = load_config(args.config)
    mcfg = cfg["diffusion_model"]["config"]
    data_train = cfg["data"]["train"]

    sample_size = data_train["sample_size"]
    h_lat = sample_size[0] // args.spatial_ratio
    w_lat = sample_size[1] // args.spatial_ratio
    tokens_per_view = args.t_lat * h_lat * w_lat

    latent_frame_rate = args.frame_rate / args.temporal_ratio
    rope_scale = (1.0 / latent_frame_rate, args.spatial_ratio, args.spatial_ratio)

    max_view = max(args.n_views)
    if args.e2e:
        # e2e uncompressed view count = 1 head + n_hands*n_fingers tactile views.
        _tac = cfg["tactile_vae"]["config"]
        _n_hands = cfg.get("projector", {}).get("num_views", 2)
        max_view = max(max_view, args.n_visual + _n_hands * _tac["num_fingers"])
    model, model_kwargs = build_transformer(cfg, max_view, dtype, device)

    if args.checkpoint:
        from utils.model_utils import load_checkpoints

        load_checkpoints(model, pretrained_ckpt=args.checkpoint)
        print(">>> loaded checkpoint (sanity-check mode, not the reported result)")

    dims_base = dict(
        batch_size=args.batch_size,
        n_view_visual=args.n_visual,
        tokens_per_view=tokens_per_view,
        in_channels=mcfg["in_channels"],
        caption_channels=mcfg["caption_channels"],
        l_text=args.l_text,
        action_chunk=data_train["action_chunk"],
        action_out_channels=mcfg["action_out_channels"],
        action_in_channels=mcfg["action_in_channels"],
        add_state=bool(cfg.get("add_state", False)),
        t_lat=args.t_lat,
        h_lat=h_lat,
        w_lat=w_lat,
        rope_interpolation_scale=rope_scale,
    )

    print(
        f">>> config={args.config}\n"
        f">>> tokens_per_view={tokens_per_view} (T={args.t_lat} H={h_lat} W={w_lat})  "
        f"action_in={mcfg['action_in_channels']} action_out={mcfg['action_out_channels']}  "
        f"add_state={dims_base['add_state']}  max_view={max_view}"
    )

    if args.decompose:
        print(">>> decompose mode: WM-forward vs action-head per step")
        for v in args.n_views:
            dims = dict(dims_base)
            dims["n_view"] = v
            r = benchmark_decompose(model, dims, args, device, dtype)
            print(
                f">>> V={v:2d} tokens={r['dit_tokens']:5d}  "
                f"WM_forward={r['wm_forward_ms']}ms (x1)  "
                f"action_step={r['action_step_ms']}ms (x{r['num_steps']})  "
                f"=> total~={r['reconstructed_total_ms']}ms  "
                f"(action share {r['action_share_pct']}%)"
            )
        return

    if args.e2e:
        gpu_name = torch.cuda.get_device_name(device)
        print(f">>> e2e mode: building LTX VAE from {args.vae_dir}")
        vae = build_vae(args.vae_dir, dtype, device)
        adapter, _ = build_adapter(cfg, dtype, device)
        e2e_rows = []
        for compressed in (True, False):
            res = benchmark_e2e(
                model, vae, adapter, cfg, dims_base, args, device, dtype, compressed
            )
            e2e_rows.append(res)
            print(
                f">>> {res['variant']:12s} V={res['n_view']:2d}  "
                f"VAE={res['vae_encode_ms']}ms fuse={res['adapter_fuse_ms']}ms "
                f"denoise={res['denoise_ms']}ms  e2e_total={res['e2e_total_ms']}ms  "
                f"alloc={res['peak_alloc_gb']}GB"
            )
        meta = dict(
            gpu=gpu_name, dtype=args.dtype, batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps, iters=args.iters,
            warmup=args.warmup, config=args.config, vae_dir=args.vae_dir,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        os.makedirs(args.out_dir, exist_ok=True)
        suffix = f"_{args.tag}" if args.tag else ""
        jp = os.path.join(args.out_dir, f"tactile_view_compression_e2e{suffix}.json")
        with open(jp, "w") as f:
            json.dump({"meta": meta, "rows": e2e_rows}, f, indent=2)
        print(f">>> wrote e2e json: {jp}")
        write_e2e_markdown(
            os.path.join(args.out_dir, f"tactile_view_compression_e2e{suffix}.md"),
            e2e_rows, meta,
        )
        return

    rows = []
    for v in args.n_views:
        dims = dict(dims_base)
        dims["n_view"] = v
        try:
            res = benchmark_view(model, dims, args, device, dtype)
            rows.append(res)
            print(
                f">>> V={v:2d} tokens={res['dit_tokens']:5d}  "
                f"alloc={res['peak_alloc_gb']}GB reserved={res['peak_reserved_gb']}GB  "
                f"total={res['total_ms']}ms (first={res['first_step_ms']} "
                f"cached={res['cached_steps_ms']})"
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f">>> V={v} OOM: {e}")
                rows.append(
                    {
                        "n_view": v,
                        "n_view_visual": args.n_visual,
                        "tokens_per_view": tokens_per_view,
                        "dit_tokens": v * tokens_per_view,
                        "oom": True,
                    }
                )
                free = getattr(torch.cuda, "empty_cache", lambda: None)
                gc.collect()
                free()
                torch.cuda.reset_peak_memory_stats()
            else:
                raise

    adapter_row = None
    if args.measure_adapter:
        # Free the DiT first so the adapter peak is measured in isolation.
        del model
        gc.collect()
        torch.cuda.empty_cache()
        adapter_row = benchmark_adapter(cfg, args, device, dtype)
        print(
            f">>> adapter fuse: {adapter_row['fuse_ms']}ms  "
            f"alloc={adapter_row['peak_alloc_gb']}GB"
        )

    gpu_name = torch.cuda.get_device_name(device)
    meta = dict(
        gpu=gpu_name,
        dtype=args.dtype,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        iters=args.iters,
        warmup=args.warmup,
        config=args.config,
        weights="random" if not args.checkpoint else args.checkpoint,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    if args.isolate_each_view:
        suffix += f"_V{args.n_views[0]}"

    json_path = os.path.join(args.out_dir, f"tactile_view_compression_results{suffix}.json")
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "rows": rows, "adapter": adapter_row}, f, indent=2)
    print(f">>> wrote json: {json_path}")

    if not args.isolate_each_view:
        md_path = os.path.join(
            args.out_dir, f"tactile_view_compression_results{suffix}.md"
        )
        write_markdown(md_path, rows, adapter_row, meta)
        png_prefix = os.path.join(args.out_dir, f"tactile_view_compression{suffix}")
        maybe_plot(png_prefix, rows)


if __name__ == "__main__":
    main()
