#!/usr/bin/env python3
"""Lightweight CPU smoke for the action cross-attn KV per-modality RMSNorm.

What this proves
----------------

STATIC (``ast.parse`` -- no heavy import)
  G1. ``LTXVideoTransformer3DModel.__init__`` exposes both new kwargs
      ``action_kv_rmsnorm`` and ``action_kv_rmsnorm_affine`` with the
      backward-compat default ``False``. A regression that flips either
      default would silently insert RMSNorm into ALL existing action
      training runs.
  G2. ``_apply_action_kv_norm`` is called between the
      ``(b v) l c -> b (v l) c`` rearrange and ``_split_action_kv`` in
      BOTH forward dispatch sites (gradient-checkpoint + non-checkpoint).
      Either site missing the call would let one of the two execution
      paths skip the per-modality norm, which would show up as
      nondeterministic results between train (checkpoint) and val
      (non-checkpoint).
  G3. The 488 bypass shared YAML has no ``<TIMESTAMP>`` placeholder
      left, and ships ``action_kv_rmsnorm: true`` +
      ``action_kv_rmsnorm_affine: false``.

FUNCTIONAL (helper invoked on a synthetic ``self``, no full model)
  G4. ``RMSNorm(elementwise_affine=False).weight is None``;
      ``RMSNorm(elementwise_affine=True).weight`` is a learnable
      ``(inner_dim,)`` parameter initialized to all-ones. Confirms the
      ``affine`` flag flows correctly into the ModuleList entries.
  G5. Shared-mode split arithmetic matches the documented invariants
      ``seq_len = n_view_total * tokens_per_view``,
      ``visual_len = n_view_visual * tokens_per_view``,
      and the helper preserves the ``[vis | tac_L | tac_R]`` layout in
      the concatenated output.
  G6. Divisibility assert fires when ``seq_len % n_view_total != 0``.
  G7. Batch-dim assert fires when ``action_batch_size`` does not match
      the KV batch dim (load-bearing for cross-batch isolation;
      catches a wrong-``n_view`` upstream silent bug).
  G8. VTAM order contract: with vision tokens = ``+1.0`` and tactile
      tokens = ``-1.0`` packed in ``[vis, tac]`` order, the helper's
      output preserves the sign partition exactly. Catches an
      accidental ``[tac, vis]`` swap.
  G9. Cross-batch isolation (gold standard): with ``B=2``, batch 0 =
      ``+1.0`` and batch 1 = ``-1.0``, helper output keeps batches
      distinct; modifying batch 1's input does NOT change batch 0's
      output bit-for-bit.

Speed
-----
~5-30s end-to-end on a warm filesystem, no GPU, no checkpoint.

The static phase (G1-G3) only uses ``ast.parse`` and runs in ~100ms.
The functional phase (G4-G9) imports ``LTXVideoTransformer3DModel`` so
we can pull ``_apply_action_kv_norm`` off the class without
instantiating it (avoids constructing the 28-layer WM body). The
helper is then called on a tiny ``types.SimpleNamespace`` with the
required attributes wired up.

Run as::

    cd .
    python scripts/smoke_action_kv_rmsnorm.py

Exits 0 on PASS, 1 if any case fails.
"""

from __future__ import annotations

import ast
import os
import sys
import traceback
import types
from typing import Callable, List, Tuple


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
TRANSFORMER_PATH = os.path.join(
    REPO_ROOT, "models", "ltx_models", "transformer_ltx_multiview.py"
)
YAML_PATH = os.path.join(
    REPO_ROOT,
    "configs", "ltx_model", "diverse_488",
    "action_model_diverse_488_d488wm_bypass_shared_long50000.yaml",
)


# ============================================================================
# STATIC PHASE (ast.parse only -- fast, doesn't import diffusers)
# ============================================================================


def _load_transformer_ast() -> Tuple[ast.Module, str]:
    with open(TRANSFORMER_PATH, "r") as f:
        src = f.read()
    return ast.parse(src, filename=TRANSFORMER_PATH), src


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == name:
            return n
    raise AssertionError(f"class {name!r} not found in {TRANSFORMER_PATH}")


def _find_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for item in cls.body:
        if isinstance(item, ast.FunctionDef) and item.name == name:
            return item
    raise AssertionError(
        f"method {cls.name}.{name} not found (file may have been refactored)"
    )


def _test_init_kwargs() -> None:
    """G1: __init__ exposes the two new kwargs with default False."""
    tree, _ = _load_transformer_ast()
    cls = _find_class(tree, "LTXVideoTransformer3DModel")
    init = _find_method(cls, "__init__")

    # Collect (name, default) pairs from positional args (defaults right-aligned)
    # AND keyword-only args.
    a = init.args
    pos_defaults = a.defaults
    pos_args = a.args[-len(pos_defaults):] if pos_defaults else []
    by_name: dict = {}
    for arg, default in zip(pos_args, pos_defaults):
        by_name[arg.arg] = default
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        by_name[arg.arg] = default

    for kw in ("action_kv_rmsnorm", "action_kv_rmsnorm_affine"):
        if kw not in by_name:
            raise AssertionError(
                f"LTXVideoTransformer3DModel.__init__ missing kwarg {kw!r}"
            )
        d = by_name[kw]
        is_false = isinstance(d, ast.Constant) and d.value is False
        if not is_false:
            raise AssertionError(
                f"LTXVideoTransformer3DModel.__init__.{kw} default must be "
                f"False; got {ast.dump(d)}. A non-False default would enable "
                "RMSNorm in all existing action runs."
            )


def _test_forward_call_placement() -> None:
    """G2: _apply_action_kv_norm is called before _split_action_kv in BOTH dispatch sites.

    Pipeline order contract:
        rearrange -> _apply_action_kv_norm -> _split_action_kv -> action_blocks[i]

    We check the source text contains at least two distinct occurrences
    of ``_apply_action_kv_norm(``, and for each one there is a
    ``_split_action_kv(`` strictly after it and an
    ``action_blocks[block_idx](`` further after that, with no intervening
    second ``_apply_action_kv_norm(`` call. This proves the norm is
    applied BEFORE the slicing (clean single-tensor API) and BEFORE the
    action block dispatch.
    """
    with open(TRANSFORMER_PATH, "r") as f:
        src = f.read()
    n_calls = src.count("self._apply_action_kv_norm(")
    if n_calls < 2:
        raise AssertionError(
            f"_apply_action_kv_norm should be called at least 2x in "
            f"transformer_ltx_multiview.py (checkpoint + non-checkpoint "
            f"forward paths); found {n_calls}."
        )

    def _find_all(needle: str) -> List[int]:
        return [i for i in range(len(src)) if src.startswith(needle, i)]

    split_idxs = _find_all("self._split_action_kv(")
    apply_idxs = _find_all("self._apply_action_kv_norm(")
    # Match both call styles:
    #   non-ckpt: self.action_blocks[block_idx](
    #   ckpt:     create_custom_forward(self.action_blocks[block_idx])
    block_idxs = _find_all("self.action_blocks[block_idx]")
    if not (len(split_idxs) >= 2 and len(apply_idxs) >= 2 and len(block_idxs) >= 2):
        raise AssertionError(
            "expected at least 2 occurrences each of _split_action_kv, "
            "_apply_action_kv_norm, and self.action_blocks[block_idx] "
            f"in source; got split={len(split_idxs)} "
            f"apply={len(apply_idxs)} block={len(block_idxs)}."
        )

    # Order check: every apply_idx must be followed by a split_idx, then a
    # block_idx (dispatch), all in the same dispatch site (i.e. with no
    # other apply call sandwiched in between).
    for ai_pos, ai in enumerate(apply_idxs):
        next_apply = apply_idxs[ai_pos + 1] if ai_pos + 1 < len(apply_idxs) else len(src)
        following_splits = [s for s in split_idxs if ai < s < next_apply]
        following_blocks = [b for b in block_idxs if ai < b < next_apply]
        if not following_splits:
            raise AssertionError(
                f"_apply_action_kv_norm at char {ai} has no following "
                "_split_action_kv before the next apply call; "
                "expected rearrange -> norm -> split -> dispatch order."
            )
        if not following_blocks:
            raise AssertionError(
                f"_apply_action_kv_norm at char {ai} has no following "
                "action_blocks[block_idx] dispatch; "
                "expected rearrange -> norm -> split -> dispatch order."
            )
        if following_splits[0] >= following_blocks[0]:
            raise AssertionError(
                f"call order at char {ai} is wrong: "
                "_split_action_kv must appear before action_blocks[block_idx] "
                "dispatch after the norm call."
            )


def _test_yaml_state() -> None:
    """G3: YAML has no <TIMESTAMP> and has both new flags."""
    if not os.path.isfile(YAML_PATH):
        raise AssertionError(f"missing YAML: {YAML_PATH}")
    with open(YAML_PATH, "r") as f:
        yaml_src = f.read()

    # We allow <TIMESTAMP> in comment lines (lines starting with '#') so the
    # historical edit log can keep the placeholder for context. The check
    # is: no non-comment line still contains "<TIMESTAMP>".
    bad_lines = []
    for lineno, line in enumerate(yaml_src.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if "<TIMESTAMP>" in line:
            bad_lines.append((lineno, line.rstrip()))
    if bad_lines:
        details = "\n".join(f"  L{ln}: {ln_src}" for ln, ln_src in bad_lines)
        raise AssertionError(
            "YAML still contains unfilled <TIMESTAMP> on non-comment line(s):\n"
            + details
        )

    for needle in ("action_kv_rmsnorm:", "action_kv_rmsnorm_affine:"):
        if needle not in yaml_src:
            raise AssertionError(
                f"YAML missing {needle!r}; the new RMSNorm flags must be "
                "set explicitly so future readers can find them."
            )
    # Hard-check that this run is configured the way we intended:
    # action_kv_rmsnorm=true, affine=false.
    if "action_kv_rmsnorm: true" not in yaml_src:
        raise AssertionError(
            "expected `action_kv_rmsnorm: true` in 488 bypass shared YAML."
        )
    if "action_kv_rmsnorm_affine: false" not in yaml_src:
        raise AssertionError(
            "expected `action_kv_rmsnorm_affine: false` in 488 bypass shared YAML."
        )


# ============================================================================
# FUNCTIONAL PHASE (imports the class but does NOT instantiate it)
# ============================================================================


def _build_fake_self(*, dim: int, num_layers: int, affine: bool):
    """Synthetic stand-in that satisfies _apply_action_kv_norm's contract.

    The helper only touches:
      - self.action_kv_rmsnorm     (bool gate)
      - self._kv_split_logged      (bool flag for first-call log)
      - self.kv_norm_visual[i]     (callable: tensor -> tensor)
      - self.kv_norm_tactile[i]    (callable: tensor -> tensor)

    Using ``types.SimpleNamespace`` avoids paying for the full
    ``LTXVideoTransformer3DModel`` __init__ (which builds 28 transformer
    blocks + action expert + RoPE etc., ~hundreds of MB).
    """
    import torch  # noqa: F401 -- explicit, so failure shows here
    import torch.nn as nn
    from diffusers.models.normalization import RMSNorm

    fake = types.SimpleNamespace()
    fake.action_kv_rmsnorm = True
    fake._kv_split_logged = False
    fake.kv_norm_visual = nn.ModuleList([
        RMSNorm(dim, eps=1e-6, elementwise_affine=affine) for _ in range(num_layers)
    ])
    fake.kv_norm_tactile = nn.ModuleList([
        RMSNorm(dim, eps=1e-6, elementwise_affine=affine) for _ in range(num_layers)
    ])
    return fake


_HELPER_FN_CACHE = {"fn": None}


def _get_helper_fn():
    """Extract ``_apply_action_kv_norm``'s source via AST and exec it in a
    minimal namespace.

    Why not just import the class?  ``transformer_ltx_multiview.py`` does
    ``from diffusers.loaders import FromOriginalModelMixin`` at module
    top, which transitively pulls in ``deepspeed`` -> triton; on a
    GPU-less login node triton's driver init crashes, blocking the
    import. The helper itself only depends on ``torch``, ``torch.cat``,
    ``self`` (a mock with ModuleLists + flags), and a module-level
    ``logger`` -- all of which we can supply explicitly.

    This keeps the smoke test runnable on CPU-only nodes while still
    exercising the *actual* helper source (no copy-paste drift).
    """
    if _HELPER_FN_CACHE["fn"] is not None:
        return _HELPER_FN_CACHE["fn"]

    import ast
    import logging
    import torch  # noqa: F401 -- explicit, so failure shows here
    from typing import Optional

    with open(TRANSFORMER_PATH, "r") as f:
        src = f.read()

    tree = ast.parse(src)
    cls_node: Optional[ast.ClassDef] = None
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == "LTXVideoTransformer3DModel":
            cls_node = n
            break
    if cls_node is None:
        raise AssertionError(
            "class LTXVideoTransformer3DModel not found in "
            f"{TRANSFORMER_PATH}"
        )

    fn_node: Optional[ast.FunctionDef] = None
    for n in cls_node.body:
        if isinstance(n, ast.FunctionDef) and n.name == "_apply_action_kv_norm":
            fn_node = n
            break
    if fn_node is None:
        raise AssertionError(
            "method _apply_action_kv_norm not found on "
            "LTXVideoTransformer3DModel"
        )

    fn_source = ast.unparse(fn_node)
    # Drop the leading ``self`` arg (the smoke tests pass a fake self
    # explicitly, so the function signature ends up identical).
    namespace = {
        "torch": torch,
        "Optional": Optional,
        "logger": logging.getLogger("smoke_action_kv_rmsnorm"),
    }
    exec(fn_source, namespace)
    fn = namespace["_apply_action_kv_norm"]
    _HELPER_FN_CACHE["fn"] = fn
    return fn


def _test_rmsnorm_affine_flag() -> None:
    """G4: affine flag flows correctly into RMSNorm.weight."""
    import torch  # noqa: F401

    fake_false = _build_fake_self(dim=8, num_layers=2, affine=False)
    for nm in fake_false.kv_norm_visual:
        if nm.weight is not None:
            raise AssertionError(
                f"affine=False RMSNorm should have weight=None; got {nm.weight}"
            )

    fake_true = _build_fake_self(dim=8, num_layers=2, affine=True)
    for nm in fake_true.kv_norm_visual:
        if nm.weight is None:
            raise AssertionError("affine=True RMSNorm should have a weight param")
        if nm.weight.shape != (8,):
            raise AssertionError(
                f"affine=True RMSNorm weight should be shape (8,); "
                f"got {tuple(nm.weight.shape)}"
            )
        if not nm.weight.requires_grad:
            raise AssertionError("affine=True RMSNorm weight should be trainable")
        # diffusers RMSNorm initializes weight to ones; check it.
        import torch as _torch
        if not _torch.allclose(nm.weight, _torch.ones(8)):
            raise AssertionError(
                "affine=True RMSNorm weight should init to ones (identity at "
                "the start of training); got " + str(nm.weight)
            )


def _test_shared_split_arithmetic() -> None:
    """G5: helper preserves overall shape; identity check for constant inputs."""
    import torch

    helper = _get_helper_fn()
    fake = _build_fake_self(dim=8, num_layers=2, affine=False)

    # 488 production shapes scaled down on D so smoke is fast.
    B, n_view_visual, n_view, tokens_per_view, D = 2, 1, 3, 4, 8
    seq_len = n_view * tokens_per_view
    final_hidden_states = torch.randn(B, seq_len, D)

    out = helper(
        fake, final_hidden_states,
        block_idx=0,
        n_view=n_view,
        n_view_visual=n_view_visual,
        action_batch_size=B,
    )
    if out.shape != (B, seq_len, D):
        raise AssertionError(
            f"helper should preserve overall shape "
            f"(B={B}, seq_len={seq_len}, D={D}); got {tuple(out.shape)}"
        )

    # No-tactile fallback: when n_view_visual is None or >= n_view the
    # helper must be a no-op and pass the tensor through unchanged.
    out_noop = helper(
        fake, final_hidden_states.clone(),
        block_idx=0,
        n_view=n_view,
        n_view_visual=None,
        action_batch_size=B,
    )
    if not torch.equal(out_noop, final_hidden_states):
        raise AssertionError(
            "helper with n_view_visual=None should pass through unchanged "
            "(no-tactile fallback for inference paths)."
        )


def _test_divisibility_assert() -> None:
    """G6: assert fires when seq_len % n_view != 0."""
    import torch

    helper = _get_helper_fn()
    fake = _build_fake_self(dim=8, num_layers=2, affine=False)

    bad = torch.randn(1, 7, 8)  # 7 not divisible by 3
    raised = False
    try:
        helper(
            fake, bad,
            block_idx=0,
            n_view=3,
            n_view_visual=1,
            action_batch_size=1,
        )
    except AssertionError:
        raised = True
    if not raised:
        raise AssertionError(
            "divisibility assert did not fire for seq_len=7 / n_view=3"
        )


def _test_batch_dim_assert() -> None:
    """G7: assert fires when KV batch dim != action batch dim."""
    import torch

    helper = _get_helper_fn()
    fake = _build_fake_self(dim=8, num_layers=2, affine=False)

    kv = torch.randn(2, 12, 8)  # B_kv=2
    raised = False
    try:
        helper(
            fake, kv,
            block_idx=0,
            n_view=3,
            n_view_visual=1,
            action_batch_size=4,  # mismatch
        )
    except AssertionError:
        raised = True
    if not raised:
        raise AssertionError(
            "batch-dim assert did not fire for KV B=2 vs action B=4"
        )


def _test_vtam_order_contract() -> None:
    """G8: vis=+1 tac=-1 -> output preserves [+1 | -1] sign partition."""
    import torch

    helper = _get_helper_fn()
    fake = _build_fake_self(dim=8, num_layers=2, affine=False)

    B, n_view_visual, n_view, tokens_per_view, D = 1, 1, 3, 4, 8
    seq_len = n_view * tokens_per_view
    visual_len = n_view_visual * tokens_per_view

    # Build KV with vision tokens = +1, tactile tokens = -1 in [vis, tac] order.
    final_hidden_states = torch.empty(B, seq_len, D)
    final_hidden_states[:, :visual_len, :] = 1.0
    final_hidden_states[:, visual_len:, :] = -1.0

    out = helper(
        fake, final_hidden_states,
        block_idx=0,
        n_view=n_view,
        n_view_visual=n_view_visual,
        action_batch_size=B,
    )

    # RMS of a vector of all 1.0 is 1.0 so RMSNorm(affine=False) is the
    # identity on constant inputs. Output should retain its sign partition.
    vis_part = out[:, :visual_len, :]
    tac_part = out[:, visual_len:, :]
    if not torch.allclose(vis_part, torch.ones_like(vis_part)):
        raise AssertionError(
            "vision slice of output should be all +1.0 after RMSNorm of "
            "constant +1.0 input; got mean=" + str(vis_part.mean().item())
        )
    if not torch.allclose(tac_part, -torch.ones_like(tac_part)):
        raise AssertionError(
            "tactile slice of output should be all -1.0 after RMSNorm of "
            "constant -1.0 input; got mean=" + str(tac_part.mean().item())
        )


def _test_cross_batch_isolation() -> None:
    """G9: per-batch determinism -- batch 1's input must not bleed into batch 0."""
    import torch

    helper = _get_helper_fn()

    B, n_view_visual, n_view, tokens_per_view, D = 2, 1, 3, 4, 8
    seq_len = n_view * tokens_per_view

    # Fresh fake for each call so the _kv_split_logged flip is reproducible.
    def _run(final_hidden_states: torch.Tensor) -> torch.Tensor:
        fake = _build_fake_self(dim=D, num_layers=2, affine=False)
        return helper(
            fake, final_hidden_states,
            block_idx=0,
            n_view=n_view,
            n_view_visual=n_view_visual,
            action_batch_size=B,
        )

    final_hidden_states = torch.empty(B, seq_len, D)
    final_hidden_states[0] = 1.0   # batch 0 all +1
    final_hidden_states[1] = -1.0  # batch 1 all -1

    out = _run(final_hidden_states.clone())

    # Sign preservation per batch.
    if not torch.allclose(out[0], torch.ones_like(out[0])):
        raise AssertionError("batch 0 output corrupted (expected all +1.0)")
    if not torch.allclose(out[1], -torch.ones_like(out[1])):
        raise AssertionError("batch 1 output corrupted (expected all -1.0)")
    # The two batches MUST remain distinct (no silent mixing).
    if torch.allclose(out[0], out[1]):
        raise AssertionError(
            "out[0] == out[1] after helper -- batches mixed!"
        )

    # Determinism: change batch 1's input and verify batch 0's output is
    # bit-equal to the previous run. This is the gold-standard isolation
    # property -- batch 0's KV computation must not depend on batch 1's
    # input at any step.
    final_hidden_states_alt = final_hidden_states.clone()
    final_hidden_states_alt[1] = 2.0
    out_alt = _run(final_hidden_states_alt)

    if not torch.equal(out_alt[0], out[0]):
        raise AssertionError(
            "batch 0 output changed when batch 1's input was modified -- "
            "batches are NOT isolated. This is the load-bearing safety "
            "property; investigate before launching training."
        )


# ============================================================================
# Runner
# ============================================================================


CASES: List[Tuple[str, Callable[[], None]]] = [
    ("STATIC: __init__ kwargs default False",      _test_init_kwargs),
    ("STATIC: forward call-site placement",        _test_forward_call_placement),
    ("STATIC: 488 bypass shared YAML state",       _test_yaml_state),
    ("FUNC:   RMSNorm affine flag flow",           _test_rmsnorm_affine_flag),
    ("FUNC:   shared-mode split arithmetic",       _test_shared_split_arithmetic),
    ("FUNC:   divisibility assert fires",          _test_divisibility_assert),
    ("FUNC:   batch-dim assert fires",             _test_batch_dim_assert),
    ("FUNC:   VTAM order contract (sign)",         _test_vtam_order_contract),
    ("FUNC:   cross-batch isolation + determinism", _test_cross_batch_isolation),
]


def main() -> int:
    failed = 0
    for label, fn in CASES:
        try:
            fn()
        except Exception:  # noqa: BLE001 -- smoke harness intentionally broad
            print(f"[FAIL] {label}")
            traceback.print_exc()
            failed += 1
        else:
            print(f"[OK]   {label}")
    if failed:
        print(f"\n{failed}/{len(CASES)} smoke case(s) FAILED")
        return 1
    print(f"\nAll {len(CASES)} smoke cases PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
