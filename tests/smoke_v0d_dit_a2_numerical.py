"""Phase A2 numerical-equivalence smoke for ``TactileDiTTrainer``.

Goal: prove that the A2 refactor (extracting the train-loop forward + loss
block into a shared ``_forward_loss_batch`` helper) DID NOT change the
numerical behaviour of ``training=True``. We accept the refactor only if,
on the same yaml + same seed + same batch, A1 commit and A2 commit produce
bit-identical (or <= 1e-6) ``loss`` / ``loss_video`` / ``loss_visual`` /
``loss_tactile`` / ``loss_action``.

How it works:

  * In ``--capture`` mode the script builds a real ``TactileDiTTrainer``
    from the supplied yaml, forces a deterministic configuration overlay
    (seed=42, no color jitter, no caption dropout, batch_size=2,
    num_workers=0, deepspeed off), and monkey-patches
    ``accelerator.backward`` so that on the FIRST call it inspects the
    calling ``train()`` frame's locals to extract every loss component
    that ends up in that scope, dumps them to JSON, and raises a tagged
    ``RuntimeError`` to abort training before backward / optimizer step.
    Nothing is written to disk except the JSON.

  * In ``--compare`` mode the script reads two JSONs and diffs each loss
    component. Returns 0 iff every key matches within ``--atol``.

The monkey-patch works identically at A1 commit (where the forward + loss
runs inline inside ``train()``) and at A2 commit (where the forward + loss
runs inside ``_forward_loss_batch`` but the resulting losses are still
assigned as locals in the ``train()`` frame, via the
``out = self._forward_loss_batch(batch, training=True); loss = out['loss']
...`` block). So a single script captures both branches cleanly.

The yaml used should exercise the most-used train()-time code paths.
``video_model_right_hand_pick_cube_tactile_dryrun.yaml`` is recommended
because it has ``use_tactile_views=true`` and ``lambda_visual = lambda_
tactile = 1.0``, so the helper's per-modality split is exercised. Note:
this yaml uses v0c-A (no pose injection), so the v0d ``hand_pose``
branch of the helper is NOT covered by this numerical test. Coverage of
that branch falls back to (a) the static AST audit in
``smoke_v0d_dit_a2.py``, (b) the encode-level e2e check in
``smoke_v0d_dit_a1.py``, and (c) line-by-line review showing the v0d
lines in the helper are a verbatim copy from A1's inline block.

Recommended workflow (single GPU; do this BEFORE committing A2):

    # 0. Stage the smoke script outside the git tree so it survives
    #    checkouts.
    cp ./scripts/smoke_v0d_dit_a2_numerical.py \
       ./smoke_v0d_dit_a2_numerical.py

    # 1. Capture at A2 (current branch tip) FIRST -- this also serves as
    #    a smoke that the helper at least RUNS without exceptions.
    CUDA_VISIBLE_DEVICES=0 python ./smoke_v0d_dit_a2_numerical.py \
        --capture \
        --config configs/cube_handover/stage2_world_model.yaml \
        --output /tmp/a2_loss.json

    # 2. Stash and check out A1 commit
    cd .
    git stash --include-untracked
    git checkout 2e5dcab

    CUDA_VISIBLE_DEVICES=0 python ./smoke_v0d_dit_a2_numerical.py \
        --capture \
        --config configs/cube_handover/stage2_world_model.yaml \
        --output /tmp/a1_loss.json

    # 3. Restore A2 + diff
    git checkout master
    git stash pop

    python ./smoke_v0d_dit_a2_numerical.py \
        --compare /tmp/a1_loss.json /tmp/a2_loss.json --atol 1e-6

Exit code 0 from --compare means the refactor is numerically clean.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from typing import Dict


def _ensure_project_root_on_path(start_dir: str | None = None) -> str:
    """Walk up from this script until we find a dir that contains ``main.py``.

    We add that dir to ``sys.path`` so ``runner.tactile_dit_trainer`` is
    importable in --capture mode. Works whether the script lives inside
    the repo tree or has been copied elsewhere and run from the repo root.
    """
    candidates = []
    here = start_dir or os.path.dirname(os.path.abspath(__file__))
    # First try: parent dir contains main.py (this is the in-tree case).
    candidates.append(os.path.dirname(here))
    # Second try: the current working directory, for a copied-out script
    # invoked from the repo root.
    candidates.append(".")
    for cand in candidates:
        if cand and os.path.isfile(os.path.join(cand, "main.py")):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return cand
    raise RuntimeError(
        "Could not find DexVTAM project root (no main.py found in any of "
        f"{candidates!r}). Pass --project-root explicitly."
    )


def _override_config_for_smoke(
    config_path: str,
    *,
    seed: int,
    batch_size: int,
    train_steps_cap: int,
) -> str:
    """Load yaml, override knobs for deterministic 1-step capture, save tmp.

    What we override and why:

      * ``seed``                       -> fixed value (caller-supplied).
      * ``use_deepspeed: false``       -> the DeepSpeed launch path
                                          requires torchrun; we want plain
                                          ``python smoke.py`` so the
                                          accelerator runs as a 1-process
                                          local Accelerator and the train
                                          loop runs deterministically. The
                                          forward + loss math is identical
                                          either way.
      * ``batch_size: 2``              -> 1-step capture; large batch wastes
                                          mem and slows trainer init.
      * ``dataloader_num_workers: 0``  -> avoid worker-pool nondeterminism
                                          in the data iteration order.
      * ``persistent_workers: false``  -> harmless with num_workers=0 but
                                          we set it anyway.
      * ``caption_dropout_p: 0.0``     -> remove the only RNG-driven branch
                                          inside the training path; the
                                          two captures match bit-for-bit.
      * ``use_color_jitter: false``    -> ditto: color jitter also reads
                                          from the global RNG.
      * ``gradient_accumulation_steps: 1``
      * ``lr_warmup_steps: 0``
      * ``steps_to_save / log / val``  -> push way above train_steps_cap so
                                          they never fire during the run.
      * ``train_steps`` / ``optim.train_steps`` -> cap (we abort after 1).

    Note: caption_dropout AND color_jitter are training-time-only knobs
    that the A2 helper gates via the ``training`` argument. Disabling
    them here removes the SOURCE of any train/val divergence; it does
    NOT mean we're not testing A2 against A1 -- both A1 and A2 honor
    the same flag values, so disabling them cleanly removes the RNG
    contribution from BOTH commits.
    """
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["seed"] = int(seed)
    cfg["use_deepspeed"] = False
    cfg["batch_size"] = int(batch_size)
    cfg["dataloader_num_workers"] = 0
    cfg["persistent_workers"] = False
    cfg["prefetch_factor"] = None  # ignored when num_workers=0
    cfg["caption_dropout_p"] = 0.0
    cfg["use_color_jitter"] = False
    cfg["gradient_accumulation_steps"] = 1
    cfg["lr_warmup_steps"] = 0

    cfg["train_steps"] = int(train_steps_cap)
    if isinstance(cfg.get("optim"), dict):
        cfg["optim"]["train_steps"] = int(train_steps_cap)

    cfg["steps_to_save"] = int(train_steps_cap * 1000)
    cfg["steps_to_log"]  = int(train_steps_cap * 1000)
    cfg["steps_to_val"]  = int(train_steps_cap * 1000)

    # Redirect output_dir to /tmp so we don't pollute the production tree
    # (some prep code may make_dirs even before we abort).
    cfg["output_dir"] = os.path.join(
        tempfile.gettempdir(), "smoke_v0d_dit_a2_numerical_output",
    )

    tmp_path = tempfile.mktemp(prefix="smoke_v0d_dit_a2_", suffix=".yaml")
    with open(tmp_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return tmp_path


def _force_global_determinism(seed: int) -> None:
    """Set every global RNG we can reach. Called twice: once at process
    startup AND once right before train() is invoked (after trainer init
    burns some RNG)."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_dist_initialized() -> None:
    """The trainer's ``__init__`` (around line 622 in current master) calls
    ``dist.broadcast(folder_len_tensor, src=0)`` to share the run's save
    folder name across ranks. That requires the default process group to
    already be initialized -- which torchrun normally does for us, but
    we are NOT running under torchrun (intentional: we want a clean
    1-process deterministic capture).

    Workaround: build a 1-process group ourselves with localhost rendezvous
    so the trainer thinks it is a single-rank torchrun. ``dist.broadcast``
    then becomes a no-op (rank 0 -> rank 0).
    """
    import torch
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # Pick a port that is unlikely to collide with anything else on the
    # box; offset by PID so back-to-back captures don't reuse the same
    # tcpstore (which can hang briefly after the previous process exits).
    os.environ.setdefault("MASTER_PORT", str(29500 + (os.getpid() % 500)))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=0, world_size=1)


def _capture_one_step(config_path: str, seed: int) -> Dict[str, float]:
    """Build trainer, monkey-patch backward, run train(), return capture."""
    _ensure_project_root_on_path()
    _force_global_determinism(seed)
    _ensure_dist_initialized()

    import torch
    from utils import import_custom_class

    Runner = import_custom_class(
        "TactileDiTTrainer", "runner/tactile_dit_trainer.py",
    )

    runner = Runner(config_path)
    runner.prepare_dataset()
    runner.prepare_models()
    runner.prepare_trainable_parameters()
    runner.prepare_optimizer()
    runner.prepare_for_training()
    runner.prepare_trackers()

    # ---- Projector dtype fixup for single-GPU no-DeepSpeed mode ------
    # In production, the trainer is launched under DeepSpeed which
    # automatically casts every wrapped module to the configured
    # ``mixed_precision`` dtype (bf16). The projector is then bf16 at
    # forward time, matching the bf16 inputs that come out of the
    # frozen v0c-A tactile encoder.
    #
    # The smoke deliberately runs with ``use_deepspeed=false`` (single
    # process, deterministic) so the auto-cast does NOT happen, and
    # ``prepare_for_training`` registers the projector with the
    # accelerator as a no-op (no DDP, no DeepSpeed). The projector
    # weights remain in their construction dtype (fp32, see
    # ``_stage2_prepare_tactile`` line ~950), and the first
    # ``self.projector(tac_latent_pre)`` call crashes with
    # ``expected scalar type BFloat16 but found Float`` because
    # ``tac_latent_pre`` is bf16.
    #
    # Fix: do exactly what DeepSpeed would have done -- cast the
    # projector (and, for symmetry, the tactile_vae) to weight_dtype
    # after ``prepare_for_training`` has had its chance to wrap them.
    # This is applied IDENTICALLY in both the A1 and A2 captures, so
    # it does not bias the numerical comparison.
    import torch
    weight_dtype = runner.state.weight_dtype
    if runner.projector is not None and weight_dtype != torch.float32:
        runner.projector.to(dtype=weight_dtype)
        print(
            f"[smoke] cast projector to {weight_dtype} "
            f"(smoke-only; production gets this from DeepSpeed)"
        )
    if (
        getattr(runner, "tactile_vae", None) is not None
        and weight_dtype != torch.float32
    ):
        runner.tactile_vae.to(dtype=weight_dtype)

    # Re-seed RIGHT BEFORE train() so any RNG drift from the prepare_*
    # chain (dataloader sampler init, model from_pretrained, accelerate
    # prepare, ...) does NOT leak into the captured step. Both A1 and
    # A2 captures run with the SAME post-prepare RNG state because we
    # explicitly reseed here.
    _force_global_determinism(seed)

    captured: Dict[str, float] = {}
    sentinel = "__SMOKE_HALT_A2_NUMERICAL__"

    original_backward = runner.state.accelerator.backward

    def patched_backward(loss, *a, **kw):
        # Find the train() frame and grab the per-modality loss locals.
        # This works for both A1 (inline) and A2 (helper) because in
        # both cases the per-modality losses end up bound as locals in
        # the `train()` frame just above the `accelerator.backward(loss)`
        # call.
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code.co_name == "train":
                locs = frame.f_locals
                for key in (
                    "loss",
                    "loss_video",
                    "loss_visual",
                    "loss_tactile",
                    "loss_action",
                ):
                    v = locs.get(key)
                    if isinstance(v, torch.Tensor):
                        captured[key] = float(v.detach().cpu().float().item())
                    elif isinstance(v, (int, float)):
                        captured[key] = float(v)
                break
            frame = frame.f_back
        # Abort the train loop before backward; we just want the FIRST
        # step's losses.
        raise RuntimeError(sentinel)

    runner.state.accelerator.backward = patched_backward

    try:
        runner.train()
    except RuntimeError as e:
        if sentinel not in str(e):
            raise

    # Restore in case the trainer has any cleanup that runs after train().
    runner.state.accelerator.backward = original_backward
    return captured


def _run_val_smoke(config_path: str, seed: int) -> Dict[str, float]:
    """End-to-end smoke for ``TactileDiTTrainer._compute_val_loss``.

    What this proves about Phase A2 BEYOND the static AST audit and the
    numerical capture mode:

      1. ``_compute_val_loss`` actually RUNS to completion on a real
         dataloader. The capture mode only exercises
         ``_forward_loss_batch(training=True)``; bugs that only surface
         under ``training=False`` (eval-mode interactions, no_grad,
         per-rank reduce, ...) would slip through.

      2. The Gate-2 ``training=False`` log line is emitted (proves the
         val branch of the helper is reachable and the
         ``_v0d_gate2_val_logged`` one-shot flag works).

      3. RNG state is snapshot + restored correctly. We capture the
         full CPU and per-CUDA-device RNG state immediately before the
         call and assert byte-equality immediately after. If the
         try/finally restore block ever regresses, this smoke crashes
         loudly.

      4. The model is put back in train() mode on exit (the bug we are
         hedging against here is the case where val raises an exception
         mid-loop and the finally block silently leaves the model in
         eval mode, which would then silently degrade subsequent train
         steps).

      5. The return dict has every key the train loop will key off
         when it integrates the val helper in Phase A3 (loss_video,
         loss_visual, loss_tactile, loss_action, plus num_batches).

      6. For ``train_mode='video_only'`` yamls, ``loss_action == 0``
         and ``loss_video > 0``; for ``train_mode='all'``, the action
         loss must be > 0. This catches a future refactor accidentally
         activating the action branch under video_only, or vice versa.

      7. *v0d mode autodetect*: if the dataset batch carries a non-None
         ``hand_pose`` key (i.e. the yaml turned on ``read_hand_pose``),
         we additionally assert ``_v0d_gate1_encode_logged is True``
         after the val call. That proves the val codepath actually went
         through ``_encode_tactile_split`` with v0d kwargs -- not the
         v0c-A fallback. Without this check, a yaml could silently
         degrade to v0c-A inside val and the gen-gap study would be
         wrong (val running pure visual, train running v0d-fused).

      8. *Determinism (no train-side augmentation leaks into val)*:
         we call ``_compute_val_loss`` a SECOND time on the same
         dataloader / config / state, and assert the five loss scalars
         match the first call to within 1e-6. The val helper already
         snapshots + restores RNG and fixes ``val_seed``, so two
         back-to-back calls must produce bit-equal losses. If a
         train-only random op (caption dropout, color jitter, dropout
         under training=True path, attention dropout under .train()...)
         ever fires under ``training=False``, this assertion catches it.

    NOTE: This smoke uses ``runner.train_dataloader`` as a stand-in
    val dataloader. That's fine because we don't care about the loss
    VALUE here -- only that the call mechanics + determinism are
    correct. Real val dataloaders (488-episode-level holdout + cube
    val split) are wired in Phase A3.
    """
    _ensure_project_root_on_path()
    _force_global_determinism(seed)
    _ensure_dist_initialized()

    import torch
    from utils import import_custom_class

    Runner = import_custom_class(
        "TactileDiTTrainer", "runner/tactile_dit_trainer.py",
    )

    runner = Runner(config_path)
    runner.prepare_dataset()
    runner.prepare_models()
    runner.prepare_trainable_parameters()
    runner.prepare_optimizer()
    runner.prepare_for_training()
    runner.prepare_trackers()

    # Same projector dtype fixup as capture mode (see comment in
    # _capture_one_step for why this is needed in single-GPU mode).
    weight_dtype = runner.state.weight_dtype
    if runner.projector is not None and weight_dtype != torch.float32:
        runner.projector.to(dtype=weight_dtype)
    if (
        getattr(runner, "tactile_vae", None) is not None
        and weight_dtype != torch.float32
    ):
        runner.tactile_vae.to(dtype=weight_dtype)

    # ---- v0d autodetect: peek at one batch BEFORE val -------------------
    # We need to know whether this yaml is wired for v0d (the dataset
    # emits hand_pose) so we can decide whether to assert the v0d
    # encode-path Gate-1 flag after the val call. We do this with a
    # one-shot iter on train_dataloader -- the dataloader is an
    # iterable (NOT a one-shot iterator), so a fresh iter() inside
    # _compute_val_loss later still yields the same first batch.
    peek = next(iter(runner.train_dataloader))
    hand_pose = peek.get("hand_pose") if isinstance(peek, dict) else None
    has_hand_pose = (
        hand_pose is not None
        and hasattr(hand_pose, "shape")
        and hand_pose.numel() > 0
    )
    if has_hand_pose:
        v0d_mode = True
        try:
            hp_shape = tuple(hand_pose.shape)
        except Exception:
            hp_shape = "<unknown>"
        print(
            f"[val-smoke] detected v0d mode "
            f"(batch has non-empty 'hand_pose', shape={hp_shape}); "
            f"will additionally assert v0d encode Gate-1 flag is set."
        )
    else:
        v0d_mode = False
        print(
            "[val-smoke] detected v0c-A mode "
            "(batch has no 'hand_pose'); skipping v0d encode-flag assertion."
        )
    del peek, hand_pose

    # Reset RNG: the iter() above consumed some RNG (dataloader
    # workers seed off the global generator). Without this reseed the
    # two _compute_val_loss calls below could start from slightly
    # different states (the val helper re-seeds internally, but the
    # dataloader iter is created BEFORE that internal reseed, so its
    # first-batch ordering still depends on the entry RNG state).
    _force_global_determinism(seed)

    # Pre-call state. Asserting these now means a regression is caught
    # at the smoke -- not silently swallowed by the val call's
    # try/finally.
    assert getattr(runner, "_v0d_gate2_val_logged", None) is False, (
        f"pre-call: _v0d_gate2_val_logged should be False, got "
        f"{getattr(runner, '_v0d_gate2_val_logged', None)!r}"
    )
    # In production the train loop calls _prepared_model.train() at the
    # start of each epoch. The smoke didn't go through train(), so set
    # explicitly here to lock the starting state we'll assert against.
    runner._prepared_model.train()
    was_training = runner._prepared_model.training
    assert was_training is True

    # Train mode (video_only vs all) decides which downstream
    # assertion makes sense for loss_action. Read from runner.args
    # rather than hard-coding to the cube dryrun yaml.
    train_mode = getattr(runner.args, "train_mode", "video_only")

    # Snapshot ALL four global RNG streams. Without snapshotting Python
    # `random` and numpy `np.random` here, the smoke can't catch a
    # regression where _compute_val_loss restores torch but leaks
    # random/numpy state into the caller. The first version of this
    # smoke missed those two streams, which is exactly how the v0d
    # determinism failure slipped through into a real run.
    import random as _py_random
    import numpy as _np
    rng_cpu_before = torch.get_rng_state()
    rng_cuda_before = (
        [
            torch.cuda.get_rng_state(d)
            for d in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else None
    )
    rng_py_before = _py_random.getstate()
    rng_np_before = _np.random.get_state()

    # ---- Call 1 ---------------------------------------------------------
    print(
        "[val-smoke] CALL 1: invoking runner._compute_val_loss("
        "train_dataloader, max_batches=1, tag='smoke') ..."
    )
    result = runner._compute_val_loss(
        runner.train_dataloader, max_batches=1, tag="smoke",
    )

    print("[val-smoke] CALL 1 returned dict:")
    for k, v in sorted(result.items()):
        print(f"  {k:<15} = {v}")

    # ---- Assertion 1: schema ----
    expected = {
        "loss", "loss_video", "loss_visual",
        "loss_tactile", "loss_action", "num_batches",
    }
    missing = expected - set(result.keys())
    assert not missing, (
        f"_compute_val_loss return dict is missing keys: {sorted(missing)}; "
        f"got keys: {sorted(result.keys())}"
    )

    # ---- Assertion 2: num_batches honoured ----
    assert result["num_batches"] == 1, (
        f"expected num_batches == 1 (max_batches=1 + dataloader has >= 1 batch); "
        f"got {result['num_batches']}"
    )

    # ---- Assertion 3: model mode restored ----
    assert runner._prepared_model.training == was_training, (
        f"model.training was {was_training} before _compute_val_loss; "
        f"is {runner._prepared_model.training} after -- the try/finally "
        f"restore block did not put the model back in train() mode."
    )

    # ---- Assertion 4: CPU RNG restored ----
    rng_cpu_after = torch.get_rng_state()
    assert torch.equal(rng_cpu_before, rng_cpu_after), (
        "CPU RNG state was not restored after _compute_val_loss "
        "(snapshot byte-mismatch). Val is leaking RNG into train state."
    )

    # ---- Assertion 5: CUDA RNG restored (each device) ----
    if rng_cuda_before is not None:
        for d, state_before in enumerate(rng_cuda_before):
            state_after = torch.cuda.get_rng_state(d)
            assert torch.equal(state_before, state_after), (
                f"CUDA RNG state for device {d} was not restored after "
                f"_compute_val_loss. Val is perturbing the CUDA RNG."
            )

    # ---- Assertion 5b/5c: Python + numpy RNG restored ----
    # If we miss these, the dataloader's next iter() (in the next train
    # step or in a back-to-back val call) sees a different state and
    # picks different frame indices.
    assert _py_random.getstate() == rng_py_before, (
        "Python `random` state was not restored after _compute_val_loss. "
        "Val is leaking randomness into the training loop's stdlib RNG."
    )
    np_state_after = _np.random.get_state()
    # np.random state is a 5-tuple; we compare element-wise because the
    # numpy array inside it is not amenable to == comparison.
    assert np_state_after[0] == rng_np_before[0], "np RNG type changed"
    assert _np.array_equal(np_state_after[1], rng_np_before[1]), (
        "numpy `np.random` state was not restored after "
        "_compute_val_loss. Same risk as Python random: dataloader's "
        "next iter() will pick different memory frame indices."
    )
    assert np_state_after[2:] == rng_np_before[2:], "np RNG tail changed"

    # ---- Assertion 6: Gate-2 val flag set ----
    assert runner._v0d_gate2_val_logged is True, (
        "Gate-2 _v0d_gate2_val_logged flag is still False after "
        "_compute_val_loss returned. The [v0d-gate2] first call log block "
        "either did not execute, or the flag was not set. Stage 2 audit "
        "log will not show the val branch's function id."
    )

    # ---- Assertion 7: loss values are sane ----
    for k in ("loss", "loss_video", "loss_visual", "loss_tactile", "loss_action"):
        v = result[k]
        assert isinstance(v, float), (
            f"result['{k}'] should be float (from accelerator.reduce path); "
            f"got {type(v).__name__}: {v!r}"
        )
        # Loss components are squared errors, must be non-negative.
        # NaN would also fail this check since `NaN >= 0.0` is False.
        assert v >= 0.0, f"result['{k}'] = {v} is negative or NaN."

    # ---- Assertion 8: train_mode-conditioned sanity ----
    # video_only -> action branch is dead, loss_action == 0
    # all       -> action branch fires, loss_action > 0
    # Pulling train_mode from runner.args makes this smoke yaml-agnostic
    # (works for both the v0c-A and v0d cube dryrun yamls).
    if train_mode == "video_only":
        assert result["loss_action"] == 0.0, (
            f"train_mode='video_only' but loss_action={result['loss_action']} "
            f"!= 0.0; the action branch fired when it shouldn't have."
        )
        assert result["loss_video"] > 0.0, (
            f"train_mode='video_only' but loss_video={result['loss_video']} "
            f"<= 0; the visual path did not run."
        )
    elif train_mode == "all":
        assert result["loss_action"] > 0.0, (
            f"train_mode='all' but loss_action={result['loss_action']} <= 0; "
            f"the action branch did not run."
        )
        assert result["loss_video"] > 0.0, (
            f"train_mode='all' but loss_video={result['loss_video']} <= 0; "
            f"the visual path did not run."
        )
    else:
        print(
            f"[val-smoke] WARN: unrecognized train_mode={train_mode!r}; "
            f"skipping loss_action/loss_video conditional assertion."
        )

    # ---- Assertion 9: v0d encode Gate-1 flag (only when v0d_mode) ----
    if v0d_mode:
        assert getattr(runner, "_v0d_gate1_encode_logged", None) is True, (
            "v0d mode (hand_pose in batch) but _v0d_gate1_encode_logged is "
            "still False after _compute_val_loss returned. This means the "
            "val codepath did NOT enter _encode_tactile_split with v0d "
            "kwargs -- the helper silently fell back to the v0c-A path. "
            "488 midtraining val curves would be on a different forward "
            "pass than train; the gen-gap study would be invalid."
        )
        print(
            "[val-smoke] v0d assertion passed: _v0d_gate1_encode_logged=True "
            "(val path went through v0d encode with hand_pose)."
        )

    # ---- Assertion 10: determinism (Call 2 vs Call 1) ----
    # Call _compute_val_loss again with everything identical. The val
    # helper snapshots+restores RNG and seeds with a fixed val_seed,
    # so two back-to-back calls MUST produce bit-equal losses. If they
    # diverge, some train-side stochasticity (caption dropout, color
    # jitter, attention dropout under .train(), ...) is leaking into
    # the training=False codepath.
    print(
        "[val-smoke] CALL 2: re-invoking runner._compute_val_loss(...) "
        "for determinism check ..."
    )
    result2 = runner._compute_val_loss(
        runner.train_dataloader, max_batches=1, tag="smoke",
    )

    print("[val-smoke] CALL 2 returned dict:")
    for k, v in sorted(result2.items()):
        print(f"  {k:<15} = {v}")

    det_atol = 1e-6
    for k in ("loss", "loss_video", "loss_visual", "loss_tactile", "loss_action"):
        a, b = result[k], result2[k]
        diff = abs(a - b)
        assert diff <= det_atol, (
            f"determinism violated for {k}:\n"
            f"  call1 = {a:.10f}\n"
            f"  call2 = {b:.10f}\n"
            f"  |diff| = {diff:.3e}  (atol = {det_atol:.0e})\n"
            f"This means val has non-deterministic behavior even though "
            f"_compute_val_loss snapshots+restores RNG and seeds with a "
            f"fixed val_seed. Most likely cause: a train-side random op "
            f"(caption dropout / color jitter / dropout / BN running stats "
            f"/ attention dropout under .train() mode) is firing inside "
            f"_forward_loss_batch when training=False."
        )

    print(
        f"[val-smoke] determinism check passed: all 5 loss scalars "
        f"match between Call 1 and Call 2 within {det_atol:.0e}."
    )

    return result


def _print_capture_banner(captured: Dict[str, float]) -> None:
    print("[smoke] captured losses (first batch, before backward):")
    if not captured:
        print("  <empty>  -- backward was never called?")
        return
    width = max(len(k) for k in captured) + 2
    for k, v in captured.items():
        print(f"  {k:<{width}} = {v:.10g}")


def _compare(file_a: str, file_b: str, atol: float = 1e-6) -> int:
    with open(file_a) as f:
        a = json.load(f)
    with open(file_b) as f:
        b = json.load(f)

    keys = sorted(set(a.keys()) | set(b.keys()))
    all_ok = True
    print(f"[smoke] comparing {file_a} vs {file_b}  (atol={atol:.1e})")
    width = max(len(k) for k in keys) + 2
    for k in keys:
        va = a.get(k)
        vb = b.get(k)
        if va is None or vb is None:
            print(f"  {k:<{width}} MISSING  (A={va}  B={vb})")
            all_ok = False
            continue
        diff = abs(va - vb)
        ok = diff <= atol
        flag = "OK" if ok else "*** FAIL ***"
        print(
            f"  {k:<{width}} A={va:.10g}  B={vb:.10g}  diff={diff:.3e}  {flag}"
        )
        if not ok:
            all_ok = False

    if all_ok:
        print("\n[smoke PASS] every loss component matches within tolerance.")
        return 0
    print("\n[smoke FAIL] one or more loss components diverge beyond tolerance.")
    return 1


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--capture", action="store_true",
                   help="Run trainer + capture first-batch losses to --output.")
    p.add_argument("--val-smoke", action="store_true",
                   dest="val_smoke",
                   help="Run _compute_val_loss end-to-end check (no JSON output).")
    p.add_argument("--config", help="Path to a Stage-2 yaml (capture or val-smoke mode).")
    p.add_argument("--output", help="Where to write captured-loss JSON (capture mode only).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--train-steps-cap", type=int, default=100)
    p.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"),
                   help="Compare two captured-loss JSONs.")
    p.add_argument("--atol", type=float, default=1e-6)
    args = p.parse_args()

    if args.compare:
        return _compare(args.compare[0], args.compare[1], atol=args.atol)

    if args.val_smoke:
        if not args.config:
            p.error("--val-smoke requires --config")
        config_path = _override_config_for_smoke(
            args.config,
            seed=args.seed,
            batch_size=args.batch_size,
            train_steps_cap=args.train_steps_cap,
        )
        print(f"[val-smoke] overridden config -> {config_path}")
        try:
            result = _run_val_smoke(config_path, seed=args.seed)
        except AssertionError as e:
            print(f"\n[val-smoke FAIL] {e}", file=sys.stderr)
            traceback.print_exc()
            return 1
        except Exception as e:
            print(f"\n[val-smoke ERROR] unexpected exception: {e}", file=sys.stderr)
            traceback.print_exc()
            return 2
        print(
            "\n[val-smoke PASS] _compute_val_loss runs end-to-end:\n"
            "  - returns the expected schema (loss/loss_video/loss_visual/"
            "loss_tactile/loss_action/num_batches)\n"
            "  - num_batches honoured (max_batches=1 -> num_batches=1)\n"
            "  - model.training restored after the call\n"
            "  - CPU + per-CUDA-device RNG state restored after the call\n"
            "  - Gate-2 _v0d_gate2_val_logged flag flipped to True\n"
            "  - loss values are non-negative and finite; loss_action/"
            "loss_video sanity matches yaml train_mode.\n"
            "  - [v0d only] _v0d_gate1_encode_logged set -> val codepath\n"
            "    went through _encode_tactile_split with hand_pose.\n"
            "  - DETERMINISM: two back-to-back _compute_val_loss calls\n"
            "    produce bit-equal losses (|diff| <= 1e-6 on all 5 scalars).\n"
            "    No train-side augmentation / dropout leaks into val."
        )
        return 0

    if not args.capture:
        p.error("must pass exactly one of --capture, --val-smoke, --compare")
    if not args.config or not args.output:
        p.error("--capture requires --config and --output")

    config_path = _override_config_for_smoke(
        args.config,
        seed=args.seed,
        batch_size=args.batch_size,
        train_steps_cap=args.train_steps_cap,
    )
    print(f"[smoke] overridden config -> {config_path}")
    print(f"[smoke] seed={args.seed}  batch_size={args.batch_size}")

    try:
        captured = _capture_one_step(config_path, seed=args.seed)
    except Exception as e:
        print(f"[smoke ERROR] capture raised: {e}", file=sys.stderr)
        traceback.print_exc()
        return 2

    _print_capture_banner(captured)

    if not captured:
        print(
            "[smoke FAIL] captured dict is empty. The backward monkey-patch "
            "was never reached -- something aborted before the first "
            "accelerator.backward(loss). Check the trainer init log.",
            file=sys.stderr,
        )
        return 1

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(captured, f, indent=2, sort_keys=True)
    print(f"[smoke] saved capture -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
