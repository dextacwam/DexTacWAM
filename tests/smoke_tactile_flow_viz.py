#!/usr/bin/env python3
"""Lightweight smoke test for the validation tactile flow viz stack.

What this proves
----------------

1. ``utils.tactile_flow_viz.save_flow_compare_grid`` handles both the
   Stage-1 single-hand input shape ``(F, H, W, C)`` and the Stage-2 WM
   multi-hand input shape ``(V_hand, F, H, W, C)``, writes a non-empty
   PNG, and rejects shape-mismatched inputs with ``ValueError``.

2. ``utils.tactile_flow_viz.resolve_frame_indices`` (shared by online
   val and the offline driver) parses ``"all"``, ``"last"``, positive
   ints, negative ints, and rejects out-of-range / non-numeric specs.

3. ``models.pipeline.custom_pipeline.CustomPipeline.infer`` exposes the
   three tactile-injection kwargs introduced for online val:
   ``tactile_mem_latents``, ``n_view_visual``, ``n_view_tactile``. All
   three MUST default to ``None`` -- this is the contract that keeps
   every existing ``pipe.infer(...)`` caller (including training-time
   inference paths that do not pass tactile kwargs) byte-identical to
   pre-patch behavior. A regression that flips any of those defaults to
   a non-None value would silently inject tactile rows into ALL infer
   calls. We check this via ``ast``, not by importing the class, so
   the smoke does not pay for diffusers / transformers import startup.

Speed
-----
The smoke deliberately AVOIDS:

* Importing ``utils`` (whose ``__init__.py`` pulls in torchvision +
  torch.distributed). Instead we load ``utils/tactile_flow_viz.py``
  directly with ``importlib.util.spec_from_file_location``; that file
  only needs ``matplotlib`` + ``numpy`` + ``torch``.
* Importing ``CustomPipeline`` (which transitively pulls diffusers /
  transformers, ~minutes on cold filesystem). The signature contract
  is checked statically via the ``ast`` module.

End-to-end runtime: <10s on a warm filesystem, no GPU.

Run as::

    cd .
    python scripts/smoke_tactile_flow_viz.py

Exits 0 on PASS, 1 if any case fails.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import tempfile
import traceback
from typing import Callable, List, Tuple

import numpy as np


# ---- Paths (no package import; we load files directly) ----------------------
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
TACTILE_FLOW_VIZ_PATH = os.path.join(REPO_ROOT, "utils", "tactile_flow_viz.py")
CUSTOM_PIPELINE_PATH = os.path.join(
    REPO_ROOT, "models", "pipeline", "custom_pipeline.py"
)


def _load_module(path: str, name: str):
    """Load a single .py file as a standalone module.

    Bypasses package ``__init__.py`` files, which in this repo pull in
    torchvision + torch.distributed even though ``tactile_flow_viz``
    itself only needs matplotlib + numpy + torch.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # Make the module importable by its declared name during exec_module so
    # any internal "from __future__ import annotations" + later
    # ``importlib.import_module`` round-trips still resolve.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def _test_viz_single_hand() -> None:
    tfv = _load_module(TACTILE_FLOW_VIZ_PATH, "_smoke_tfv")
    rng = np.random.default_rng(0)
    F, H, W, C = 5, 24, 32, 3
    gt = rng.standard_normal((F, H, W, C)).astype(np.float32)
    pred = rng.standard_normal((F, H, W, C)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "single.png")
        ret = tfv.save_flow_compare_grid(
            flow_gt=gt,
            flow_pred=pred,
            save_path=out,
            title="single-hand smoke",
        )
        if ret != os.path.abspath(out):
            raise AssertionError(
                f"return path mismatch: {ret!r} vs {os.path.abspath(out)!r}"
            )
        if os.path.getsize(out) <= 0:
            raise AssertionError("PNG size <= 0")


def _test_viz_multi_hand() -> None:
    tfv = _load_module(TACTILE_FLOW_VIZ_PATH, "_smoke_tfv")
    rng = np.random.default_rng(1)
    V, F, H, W, C = 2, 5, 24, 32, 3
    gt = rng.standard_normal((V, F, H, W, C)).astype(np.float32)
    pred = rng.standard_normal((V, F, H, W, C)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "multi.png")
        tfv.save_flow_compare_grid(
            flow_gt=gt,
            flow_pred=pred,
            save_path=out,
            title="multi-hand smoke",
            hand_names=("left", "right"),
        )
        if os.path.getsize(out) <= 0:
            raise AssertionError("PNG size <= 0")


def _test_viz_shape_mismatch() -> None:
    tfv = _load_module(TACTILE_FLOW_VIZ_PATH, "_smoke_tfv")
    gt = np.zeros((5, 24, 32, 3), dtype=np.float32)
    pred = np.zeros((5, 24, 16, 3), dtype=np.float32)  # bad width
    raised = False
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "bad.png")
        try:
            tfv.save_flow_compare_grid(flow_gt=gt, flow_pred=pred, save_path=out)
        except ValueError:
            raised = True
    if not raised:
        raise AssertionError("expected ValueError on shape mismatch, none raised")


def _test_resolve_frame_indices() -> None:
    tfv = _load_module(TACTILE_FLOW_VIZ_PATH, "_smoke_tfv")
    fn = tfv.resolve_frame_indices

    T = 9
    cases: List[Tuple[str, List[int]]] = [
        ("all", list(range(T))),
        ("ALL", list(range(T))),
        ("last", [T - 1]),
        ("Last", [T - 1]),
        ("0", [0]),
        ("5", [5]),
        ("-1", [T - 1]),
        ("-9", [0]),
    ]
    for spec, want in cases:
        got = fn(spec, T)
        if got != want:
            raise AssertionError(
                f"resolve_frame_indices({spec!r}, {T}) == {got!r}, want {want!r}"
            )

    error_cases = ["9", "-10", "garbage", "", "1.5"]
    for spec in error_cases:
        raised = False
        try:
            fn(spec, T)
        except ValueError:
            raised = True
        if not raised:
            raise AssertionError(
                f"resolve_frame_indices({spec!r}, {T}) expected ValueError"
            )


def _test_pipeline_signature_backcompat() -> None:
    """Static check of ``CustomPipeline.infer`` signature via ``ast``.

    We do NOT import the class -- importing diffusers + transformers
    can take minutes on a cold network FS, and this smoke needs to run
    in seconds. Parsing the source is sufficient because all three
    contract pieces (kwarg present, default literal ``None``) live in
    the function header itself.
    """
    with open(CUSTOM_PIPELINE_PATH, "r") as f:
        tree = ast.parse(f.read(), filename=CUSTOM_PIPELINE_PATH)

    infer_node = None
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        if cls.name != "CustomPipeline":
            continue
        for item in cls.body:
            if isinstance(item, ast.FunctionDef) and item.name == "infer":
                infer_node = item
                break
        if infer_node is not None:
            break
    if infer_node is None:
        raise AssertionError(
            "could not find CustomPipeline.infer in "
            f"{CUSTOM_PIPELINE_PATH} (file may have been refactored)."
        )

    # ast normalises keyword-only args under args.kwonlyargs (with
    # paired defaults under args.kw_defaults); positional args live in
    # args.args (defaults in args.defaults, right-aligned). We probe
    # both bins so the contract holds regardless of which side of the
    # signature the kwargs live on.
    required = ("tactile_mem_latents", "n_view_visual", "n_view_tactile")
    a = infer_node.args
    by_name = {}
    for arg, default in zip(a.args[-len(a.defaults):], a.defaults):
        by_name[arg.arg] = default
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        by_name[arg.arg] = default

    for name in required:
        if name not in by_name:
            raise AssertionError(
                f"CustomPipeline.infer is missing tactile kwarg {name!r}; "
                "online val + offline viz will both break."
            )
        default = by_name[name]
        is_none_literal = isinstance(default, ast.Constant) and default.value is None
        if not is_none_literal:
            raise AssertionError(
                f"CustomPipeline.infer.{name} default must be None; got "
                f"{ast.dump(default)}. Non-None defaults would silently inject "
                "tactile rows into ALL callers (training-time pipe.infer "
                "included)."
            )


CASES: List[Tuple[str, Callable[[], None]]] = [
    ("save_flow_compare_grid single-hand", _test_viz_single_hand),
    ("save_flow_compare_grid multi-hand", _test_viz_multi_hand),
    ("save_flow_compare_grid shape mismatch", _test_viz_shape_mismatch),
    ("resolve_frame_indices", _test_resolve_frame_indices),
    ("CustomPipeline.infer signature backward compatibility (ast)",
     _test_pipeline_signature_backcompat),
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
