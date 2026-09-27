"""Phase A2 Gate-2 smoke: verify the shared ``_forward_loss_batch`` helper
and ``_compute_val_loss`` API on ``TactileDiTTrainer`` exist with the
expected signatures, and confirm both train and val call sites resolve to
the SAME function object (i.e. there is no accidental copy-pasted loss
logic between train and val).

What the smoke proves (Phase A2's Gate-2 checklist):

  1. ``TactileDiTTrainer._forward_loss_batch`` exists and exposes the
     expected signature ``(self, batch, *, training)``.
  2. ``TactileDiTTrainer._compute_val_loss`` exists and exposes the
     expected signature
     ``(self, val_dataloader, *, max_batches=None, tag='val')``.
  3. ``__init__`` initialises the two Gate-2 one-shot flags
     ``_v0d_gate2_train_logged`` and ``_v0d_gate2_val_logged`` to
     ``False`` (so the first train batch and the first val batch each
     emit a single proof block).
  4. The train-loop body in ``train()`` references the helper via
     ``self._forward_loss_batch(batch, training=True)`` exactly once
     (i.e. the refactor that swapped the inline forward + loss for the
     helper call landed correctly).

This smoke is INTENTIONALLY static. A2 only refactors the codepath; it
does not change tensor shapes, loss math, or model layout. A behavioural
smoke that actually instantiates a trainer + dataloader + runs one
forward+loss pass is deferred to Phase A3 (where the new val dataloader
is wired in and the integration is the natural test target). The static
checks here are enough to catch:

  * accidental signature drift (e.g. ``training`` flipped from kw-only
    to positional, breaking the train caller),
  * accidental copy-paste of loss logic into ``_compute_val_loss`` that
    no longer flows through the shared helper,
  * accidental loss of the Gate-2 init in ``__init__`` (would defeat
    the audit log).

Usage (any host, no GPU required since this only does static inspection):

    python3 scripts/smoke_v0d_dit_a2.py

Exits with code 0 on success and prints a "[smoke PASS]" banner.
"""

import ast
import inspect
import os
import re
import sys
import traceback


def _project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)  # the Genie-Envisioner checkout root


def _trainer_source_path() -> str:
    return os.path.join(_project_root(), "runner", "tactile_dit_trainer.py")


def _parse_trainer_module() -> ast.Module:
    """Parse the trainer module via :mod:`ast` so the smoke does NOT
    import the trainer (which would pull in LTX VAE, diffusers, triton,
    deepspeed, ...). Static inspection on the source is enough for A2.
    """
    src = open(_trainer_source_path(), "r").read()
    return ast.parse(src)


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name!r} not found in trainer module")


def _find_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method {name!r} not found on class {cls.name!r}")


def _arg_kinds(fn: ast.FunctionDef):
    """Return a dict describing the (kw-only / positional) split of args."""
    args = fn.args
    return {
        "posonly":  [a.arg for a in args.posonlyargs],
        "positional": [a.arg for a in args.args],
        "kwonly":   [a.arg for a in args.kwonlyargs],
        "vararg":   args.vararg.arg if args.vararg else None,
        "kwarg":    args.kwarg.arg if args.kwarg else None,
        "defaults": len(args.defaults),
        "kwonly_defaults": [d is not None for d in args.kw_defaults],
    }


def _run_signature_check(tree: ast.Module) -> None:
    """Check 1+2: the two new methods exist with the expected shape."""
    cls = _find_class(tree, "TactileDiTTrainer")

    flb = _find_method(cls, "_forward_loss_batch")
    flb_kinds = _arg_kinds(flb)
    # Expected: self, batch as positional; training as keyword-only.
    assert flb_kinds["positional"] == ["self", "batch"], (
        f"_forward_loss_batch positional args were {flb_kinds['positional']!r}, "
        f"expected ['self', 'batch']"
    )
    assert "training" in flb_kinds["kwonly"], (
        f"_forward_loss_batch must have `training` as keyword-only; got "
        f"kwonly={flb_kinds['kwonly']!r}"
    )
    # `training` itself MUST NOT have a default; the caller is required
    # to declare intent explicitly. If a future commit makes it default
    # to False, train would silently slip into "val mode" branches
    # (skipping color jitter / caption dropout). Pin it here.
    train_idx = flb_kinds["kwonly"].index("training")
    assert not flb_kinds["kwonly_defaults"][train_idx], (
        "_forward_loss_batch.training must NOT have a default; the "
        "caller must explicitly pass True (train) or False (val)."
    )
    print("[smoke] check 1 PASS: _forward_loss_batch(self, batch, *, training).")

    cvl = _find_method(cls, "_compute_val_loss")
    cvl_kinds = _arg_kinds(cvl)
    assert cvl_kinds["positional"] == ["self", "val_dataloader"], (
        f"_compute_val_loss positional args were "
        f"{cvl_kinds['positional']!r}, expected ['self', 'val_dataloader']"
    )
    assert set(cvl_kinds["kwonly"]) >= {"max_batches", "tag"}, (
        f"_compute_val_loss must expose `max_batches` and `tag` as "
        f"keyword-only; got kwonly={cvl_kinds['kwonly']!r}"
    )
    print(
        "[smoke] check 2 PASS: _compute_val_loss(self, val_dataloader, *, "
        "max_batches=None, tag='val')."
    )


def _run_init_flag_check(tree: ast.Module) -> None:
    """Check 3: __init__ initialises both Gate-2 flags to False."""
    cls = _find_class(tree, "TactileDiTTrainer")
    init = _find_method(cls, "__init__")
    src = ast.unparse(init)

    for flag in ("_v0d_gate2_train_logged", "_v0d_gate2_val_logged"):
        pat = re.compile(rf"self\.{flag}\s*=\s*False")
        assert pat.search(src), (
            f"__init__ does not set `self.{flag} = False`; the Gate-2 "
            f"first-call log relies on this initialisation."
        )
    print(
        "[smoke] check 3 PASS: __init__ initialises both _v0d_gate2_*_logged "
        "flags to False."
    )


def _run_train_call_site_check(tree: ast.Module) -> None:
    """Check 4: train() calls _forward_loss_batch(batch, training=True)
    exactly once, and does not still contain an inline forward+loss block.
    """
    cls = _find_class(tree, "TactileDiTTrainer")
    train_fn = _find_method(cls, "train")
    src = ast.unparse(train_fn)

    n_calls = len(re.findall(
        r"self\._forward_loss_batch\(\s*batch\s*,\s*training\s*=\s*True\s*\)",
        src,
    ))
    assert n_calls == 1, (
        f"expected exactly 1 call to "
        f"`self._forward_loss_batch(batch, training=True)` in train(); "
        f"found {n_calls}. Either the refactor did not land cleanly, or a "
        f"second call site sneaked in."
    )

    # Inline-block sentinel: the old forward + loss block in train()
    # contained the very characteristic `compute_density_for_timestep_sampling(`
    # call. After the refactor that call must live ONLY inside the
    # helper, not in train().
    assert "compute_density_for_timestep_sampling(" not in src, (
        "train() still contains `compute_density_for_timestep_sampling(`; "
        "the inline forward + loss block was not fully extracted into "
        "_forward_loss_batch."
    )
    # Mirror sentinel: `forward_pass(` (the DiT call) must also be gone
    # from train().
    assert "forward_pass(" not in src, (
        "train() still contains a `forward_pass(` call; the inline "
        "forward + loss block was not fully extracted."
    )
    print(
        "[smoke] check 4 PASS: train() calls the shared helper exactly once "
        "and contains no residual inline forward+loss block."
    )


def _run_compute_val_loss_uses_helper_check(tree: ast.Module) -> None:
    """Check 5 (Gate 2 hard contract): _compute_val_loss invokes
    `self._forward_loss_batch(batch, training=False)` and does NOT
    re-implement forward + loss logic.
    """
    cls = _find_class(tree, "TactileDiTTrainer")
    cvl = _find_method(cls, "_compute_val_loss")
    src = ast.unparse(cvl)

    n_calls = len(re.findall(
        r"self\._forward_loss_batch\(\s*batch\s*,\s*training\s*=\s*False\s*\)",
        src,
    ))
    assert n_calls == 1, (
        f"expected exactly 1 call to "
        f"`self._forward_loss_batch(batch, training=False)` in "
        f"_compute_val_loss(); found {n_calls}."
    )
    # If val accidentally got its own copy of the loss recipe, these
    # would show up here too.
    assert "compute_density_for_timestep_sampling(" not in src, (
        "_compute_val_loss contains `compute_density_for_timestep_sampling(`; "
        "val must delegate ALL forward+loss math to the shared helper "
        "(Gate 2 contract)."
    )
    assert "forward_pass(" not in src, (
        "_compute_val_loss contains a `forward_pass(` call; val must "
        "delegate ALL forward+loss math to the shared helper (Gate 2)."
    )
    # Must wrap iteration in `torch.no_grad()`.
    assert "torch.no_grad(" in src, (
        "_compute_val_loss must run the val loop under `torch.no_grad()`."
    )
    print(
        "[smoke] check 5 PASS: _compute_val_loss delegates to the shared "
        "helper and runs under torch.no_grad()."
    )


def main() -> int:
    print(f"[smoke] Phase A2 Gate-2 static smoke on {_trainer_source_path()}")
    try:
        tree = _parse_trainer_module()
        _run_signature_check(tree)
        _run_init_flag_check(tree)
        _run_train_call_site_check(tree)
        _run_compute_val_loss_uses_helper_check(tree)
    except AssertionError as e:
        print(f"[smoke FAIL] {e}", file=sys.stderr)
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"[smoke ERROR] unexpected exception: {e}", file=sys.stderr)
        traceback.print_exc()
        return 2
    print(
        "\n[smoke PASS] Phase A2 Gate-2 verified: shared "
        "`_forward_loss_batch` helper exists; train and val both route "
        "through it; Gate-2 flags initialised; no residual inline loss "
        "code in train() or _compute_val_loss()."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
