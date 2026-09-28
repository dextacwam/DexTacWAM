# Rollout runbook — `relative_eef_rot6d`

Two machines. The **GPU box** serves the policy; your **laptop** runs the client
and talks to the robot. Nothing here is task-specific on the client side: the
server owns the prompt, the layout and the camera list, and the client adapts to
them and verifies what it can.

Eight tasks are covered, in three shapes. The shape is the only thing that
changes for an operator; everything else is a different string on the command
line.

| shape | tasks | cameras | `action_in_channels` / `max_view` |
|---|---|---|---|
| right-only, 2 cameras | `bowl`, `tong`, `placed_tong` | head + right wrist | 113 / 3 |
| right-only, head only | `pick_place_cube` | head | 113 / **2** |
| bimanual, 3 cameras | `unscrew_v2`, `chip_with_wrist`, `cube_handover`, `wipe_white_board` | head + both wrists | 226 / 5 |

Right-only does not mean the client sends less: it sends a full bimanual
observation either way and the server trims it.

`pick_place_cube` drops the right wrist camera **by design**, so the policy has
to act through the occlusion of its own hand. Its widths are bowl's, so guard 2's
`max_view` comparison — 2 against 3 — is the only thing that separates the two
checkpoints. Do not "fix" a `max_view` mismatch by editing the yaml.

`cube_handover` and `wipe_white_board` each have a second checkpoint, the
**tactile-world-model ablation**, reached with `VARIANT=ablation`. It is the same
corpus, prompt, action space and camera set; the one difference is that its
tactile skips the world model and enters the action expert's own cross-attention
directly. The server reports which routing it loaded as `tactile_routing` in
`ping`, and the client still sends tactile identically in both.

Gate evidence for every claim below is in
[deploy_gates_right_only.md](deploy_gates_right_only.md). Launch variants for the
other checkpoint families are in [../COMMANDS.md](../COMMANDS.md).

**Status.** Bowl and `unscrew_v2` are cleared for a low-speed contact rollout
once section D passes; `unscrew_v2` passes F1, F2, F3 and the bimanual
byte-parity gate P. **`chip_with_wrist` and `placed_tong` have only F1 and F2.**
Chip is a re-test of an older corpus — [Chip](#chip) at the end says what that
costs; `placed_tong` is simply new, and its F3 and P have not been run yet.
**Tong is not cleared at all** — see [Tong](#tong).
**The three 080x tasks and both ablations have F2 only** — the dry-run passes for
each of their five configs, which fixes the layout, the stats widths, the prompt
and the compose round-trip, but no checkpoint has been loaded through F1 or
driven through the wire in F3. Run both before the robot sees any of them.

---

## A. GPU box — code

```bash
cd ~/repos/DexVTAM
git fetch origin
git checkout eef_9d_relative_deploy
git pull
git log --oneline -1              # 99e6d0f or later for unscrew_v2;
                                  # 5cf9623 or later for the 080x tasks and the ablations;
                                  # 2909142 or later to actually serve an ablation
```

The third pin is not the same kind as the first two. `5cf9623` is what *added*
`VARIANT=ablation`, but on it the server built the late-fuse tactile tokens and
never passed the hand count, so an ablation rollout died inside
`_embed_tactile_late_fuse` before the first denoise step — after the four guards
and the banner and the 4 GB load, which is exactly where you would start
suspecting the checkpoint. `2909142` is the fix. Full-model rollouts never took
that path and are unaffected.

## B. GPU box — weights

Weights live on the Hub; the config and stats stay in git. The published repo is
**flat** — the `step_N` directory's contents sit at the repo root — so download
into a directory named for the step:

```bash
source ~/repos/DexVTAM/.venv/bin/activate   # hf is installed in the venv, not system-wide
hf auth login                      # the repos are private

BOWL=~/repos/DexVTAM/outputs/stage3_action_full_0729_bowl_right_only_eef_relative_wm_bypass_shared_rmsnorm_long50000/step_10000
hf download JensenYuan/DexTacWAM_bowl --local-dir "$BOWL" --exclude "README.md"

TONG=~/repos/DexVTAM/outputs/stage3_action_full_0729_tong_right_only_eef_relative_wm_bypass_shared_rmsnorm_long50000/step_20000
hf download JensenYuan/DexTacWAM_tong --local-dir "$TONG" --exclude "README.md"

UNSCREW=~/repos/DexVTAM/outputs/stage3_action_full_0730_unscrew_bottle_cap_v2_with_wrist_eef_relative_wm_bypass_shared_rmsnorm_long50000/step_30000
hf download JensenYuan/DexTacWAM_unscrew_v2 --local-dir "$UNSCREW" --exclude "README.md"

CHIP=~/repos/DexVTAM/outputs/stage3_action_full_0718_pick_chip_with_wrist_eef_relative_chipwm_bypass_shared_rmsnorm_long50000/step_20000
hf download JensenYuan/DexTacWAM_chip_with_wrist --local-dir "$CHIP" --exclude "README.md"

PLACED=~/repos/DexVTAM/outputs/stage3_action_full_0801_placed_tong_right_only_eef_relative_wm_bypass_shared_rmsnorm_long50000/step_20000
hf download JensenYuan/DexTacWAM_placed_tong_3w_2w --local-dir "$PLACED" --exclude "README.md"

ls -la "$BOWL"    # config.json, diffusion_pytorch_model.safetensors (~4.2G), projector.pt
```

The 080x tasks and the ablation follow the same form. Unlike the five above they
were published one at a time, at whatever step their B3 had reached, so the
steps are not uniform — the training run each was cut from is named beside it:

```bash
# from run 2026_08_08_16_04_26
HANDOVER=~/repos/DexVTAM/outputs/stage3_action_full_0806_cube_handover_with_wrist_eef_relative_wm_bypass_shared_rmsnorm_long50000/step_10000
hf download JensenYuan/DexTacWAM_cube_handover --local-dir "$HANDOVER" --exclude "README.md"

# from run 2026_08_09_20_13_07
WIPE=~/repos/DexVTAM/outputs/stage3_action_full_0808_wipe_white_board_with_wrist_eef_relative_wm_bypass_shared_rmsnorm_long50000/step_10000
hf download JensenYuan/DexTacWAM_wipe_white_board --local-dir "$WIPE" --exclude "README.md"

# from run 2026_08_11_07_07_06
PICKPLACE=~/repos/DexVTAM/outputs/stage3_action_full_0809_pick_place_cube_eef_relative_wm_bypass_shared_rmsnorm_long50000/step_20000
hf download JensenYuan/DexTacWAM_pick_place_cube --local-dir "$PICKPLACE" --exclude "README.md"

# tactile-WM ablation: note the DIFFERENT run root, which is what VARIANT=ablation
# resolves. From run 2026_08_11_01_04_04.
WIPE_ABL=~/repos/DexVTAM/outputs/stage3_action_full_0808_wipe_white_board_tacwm_ablation_late_fuse_long50000/step_10000
hf download JensenYuan/DexTacWAM_wipe_white_board_tacwm_ablation --local-dir "$WIPE_ABL" --exclude "README.md"

# From run 2026_08_12_01_40_43. Note the repo name is NOT parallel to wipe's:
# no `tacwm_` in it.
HANDOVER_ABL=~/repos/DexVTAM/outputs/stage3_action_full_0806_cube_handover_tacwm_ablation_late_fuse_long50000/step_10000
hf download JensenYuan/DexTacWAM_cube_handover_ablation --local-dir "$HANDOVER_ABL" --exclude "README.md"
```

`cube_handover`'s ablation goes under its own run root,
`stage3_action_full_0806_cube_handover_tacwm_ablation_late_fuse_long50000`, which
is recorded here because the run root is the part that cannot be guessed from the
task name. Its B3 was stopped once `step_10000` was written, so that step is the
only one this run root holds.

Each Hub repo holds exactly one step, so the step number is a property of the
repo rather than of the download. Bowl, `cube_handover` and `wipe_white_board`
(full and ablation alike) are `step_10000`; tong, `chip_with_wrist`,
`placed_tong` and `pick_place_cube` `step_20000`; `unscrew_v2` `step_30000`.
Naming the directory anything else makes `STEP` describe something untrue.

Each 080x ablation sits at the same step as its full sibling — `step_10000` for
both `cube_handover` and `wipe_white_board` — by design, not coincidence: the
ablation only answers "is tactile world modelling necessary" if the two differ in
nothing but where tactile enters, and training length is the easiest of those to
lose.

An ablation and its full sibling are *not* interchangeable downloads even though
their widths match. They differ in `max_view` (3 vs 5) and in whether the
checkpoint carries the action expert's second, tactile-only cross-attention, so
crossing them fails guard 2 — which is the point of publishing them under
separate repo ids and separate run roots rather than as two steps of one run.

The directory names are not free-form either. `run_server` derives the run root
from the config stem, so the parent directory must be
`outputs/stage3_action_full_<config stem minus "action_model_">` for `STEP` to
resolve. Publishing side, for whoever produced these:

```bash
python scripts/publish_action_ckpt_to_hf.py \
  --ckpt-dir outputs/stage3_action_full_0730_unscrew_bottle_cap_v2_with_wrist_eef_relative_wm_bypass_shared_rmsnorm_long50000/<TS>/step_30000 \
  --config configs/bottle_cap/stage3_action_expert.yaml \
  --repo-id JensenYuan/DexTacWAM_unscrew_v2 \
  --task-title "Unscrew the cap off a bottle (bimanual)" \
  --private --dry-run        # drop --dry-run to upload
```

It uploads weights only and prints everything it is skipping first, so optimizer
and DeepSpeed shards cannot ship by accident. Run it from a checkout whose HEAD
is **pushed**: the generated model card pins that commit and links the config and
`stat_file` at it, and an unpushed commit makes both links 404.

Check the shared tactile adapter before loading 4 GB:

```bash
sha256sum ~/repos/vtam_shared_ckpts/tactile_vae_ckpt/model.pt   # 031d5e05...
```

This is the guard most likely to fail on a box that has not served this family.
The adapter loads with `strict=False`, so a wrong snapshot loads without any
error and shows up only as poor contact behaviour — the one failure here you
would not diagnose during a rollout.

## C. GPU box — dry-run, then serve

```bash
python web_infer_scripts/dryrun_relative_server.py \
  -c configs/bowl_unstack/stage3_action_expert.yaml \
  --task bowl
# expect: [dryrun] ALL CHECKS PASSED     (no GPU needed)

TASK=bowl STEP=10000 bash web_infer_scripts/run_server_tactile_relative.sh
# GPU=0 TASK=bowl STEP=10000 bash ...    to force a device
```

For the bimanual task, same script, different two variables:

```bash
python web_infer_scripts/dryrun_relative_server.py \
  -c configs/bottle_cap/stage3_action_expert.yaml \
  --task unscrew_v2

TASK=unscrew_v2 STEP=30000 bash web_infer_scripts/run_server_tactile_relative.sh

# chip, same shape again
TASK=chip_with_wrist STEP=20000 bash web_infer_scripts/run_server_tactile_relative.sh

# placed_tong: right-only, like bowl and tong
TASK=placed_tong STEP=20000 bash web_infer_scripts/run_server_tactile_relative.sh
```

The 080x tasks and the ablations, same script again:

```bash
TASK=cube_handover    STEP=10000 bash web_infer_scripts/run_server_tactile_relative.sh
TASK=wipe_white_board STEP=10000 bash web_infer_scripts/run_server_tactile_relative.sh

# head-only right arm; the banner should say cameras ['head_img'] and max_view=2
TASK=pick_place_cube  STEP=20000 bash web_infer_scripts/run_server_tactile_relative.sh

# tactile-WM ablation: same TASK, so the same registry prompt and the same domain
# assertion; VARIANT is the only thing that changes which config resolves
VARIANT=ablation TASK=cube_handover    STEP=10000 bash web_infer_scripts/run_server_tactile_relative.sh
VARIANT=ablation TASK=wipe_white_board STEP=10000 bash web_infer_scripts/run_server_tactile_relative.sh
```

Running without `STEP` lists the `step_*` dirs it can see and exits, which is the
quickest way to find out what a fresh download actually contains.

The ablation prints an extra line before the guards, and `ping` reports
`tactile_routing=late_fuse`. If you launched `VARIANT=ablation` and see
`world_model_views`, you are serving the full checkpoint under the ablation's
name — nothing downstream will notice, because both consume tactile and both
produce plausible motion.

`tong` and `placed_tong` share a rig, a policy and a scene; only the starting
condition differs, in that `placed_tong`'s tongs are already held. Both prompts
therefore read as reasonable for either corpus, and the banner will not save you
— it is the registry's domain assertion, which refuses a task whose expected
corpus is not the one the yaml serves, that keeps them apart.

Four guards run before the model loads: the checkpoint directory is complete,
its `config.json` agrees with the YAML on `action_in_channels` and `max_view`
(113/3 right-only two-camera, 113/2 head-only, 226/5 bimanual, 226/3 for a
bimanual ablation whose tactile is not in the view stack — this is what rejects a
checkpoint of the wrong layout), the v0d sha matches, and the dry-run passes.
Read the banner:

| expect | why it matters |
|---|---|
| `CONFIG=configs/<task>/stage3_action_expert.yaml` | the task resolved to the config you meant |
| `arm_layout=right_only`, `arms=['right']` — or `bimanual`, `['left','right']` for `unscrew_v2` | the wrong layout would mis-slice every field |
| `action_mode=relative_eef_rot6d` | absolute 4x4 EEF targets, not joint targets |
| `task=bowl`, sha `cb09decd…` | the prompt the corpus was converted with |
| `tactile_health_mode=client_30hz` | the server verifies health, it does not compute it |

The blank-fill windows printed under the banner are **per task**, not a global
constant: `tong`, `bowl` and `pick_place_cube` fill up to 5 frames on the three
operating fingers and 10 elsewhere, while `unscrew_v2`, `cube_handover` and
`wipe_white_board` fill 5 everywhere because all ten of their fingers operate.
They must match what the corpus was converted with, and the server now
reads them off the task registry entry so they cannot drift apart.
`chip_with_wrist` also shows 5 everywhere, but see [Chip](#chip) — for that one
the number is a choice rather than a reading.

Ready at `listening on 0.0.0.0:5008 — ready.` The port opens only after the
model is loaded, so a successful connect is itself part of the check.

Stop, or switch task:

```bash
pkill -f tactile_server_sharpa_dexmate
TASK=tong STEP=20000 bash web_infer_scripts/run_server_tactile_relative.sh
```

One server per port. Two at once (`PORT=5009`) is possible but reintroduces the
wrong-connection risk, whose only defence is reading the task line the client
prints.

## D. Laptop — client

### D1. Code and offline checks

```bash
cd ~/Projects/DexVTAM/dexvtam_hardware_infra
git fetch origin
git checkout feat/relative-eef-deploy
git pull
conda activate sharpa-dexmate

python teleop/tests/test_request_gate.py           # both must PASS
python teleop/tests/test_tactile_health_client.py

python teleop/dryrun_relative_eef_deploy.py        # IK convention, no server
```

Run the two tests here rather than trusting a run elsewhere: they are what
catches a stale checkout or a different numpy on this machine, and they take two
seconds.

The third needs the pinocchio/pink IK stack but no server, no arms, no hands and
no cameras. It FKs the default joints to an anchor pose, feeds absolute 4x4
targets back through the same `ArmIKManager` the client uses, and measures the
residual — which is what confirms `get_arm_action` consumes **absolute** poses in
the same base frame the server composes them in. Get that convention wrong and
nothing raises: the arm moves smoothly to the wrong place. Run it again with
`--server` once the GPU box is up — see the end of §D3.

### D2. Robot environment

```bash
cd ~/Projects/DexVTAM/dexvtam_hardware_infra
source setup.sh
echo $ROBOT_NAME                  # dm/vgd1262ab823-1p
```

From the **repo root**, not `teleop/`. Without it `Robot()` raises
`Variant not specified and neither ROBOT_CONFIG nor ROBOT_NAME ... are set`,
which is the first hardware call in `main()` and has nothing to do with the
policy. `setup.sh` also frees tactile ports 50001/50002, so skipping it produces
a second, more confusing failure later when the Sharpa hands cannot bind.

One-time sanity check on a new laptop:

```bash
python -c "import qpsolvers; print(qpsolvers.available_solvers)"   # must list 'daqp'
```

Without `daqp` the arms freeze with no error message — which during a rollout
looks exactly like a policy hold, and is the reason to check it now rather than
diagnose it later.

### D3. Camera streamers (separate terminals, leave running)

Head camera:

```bash
ssh dexmate-nano
cd zed_stream/ && sudo ./build/zed_streamer --clean --jpeg-quality 100 \
    --max-fps 30 --resolution HD1080 --no-right --no-depth --no-pc --no-imu
```

Wrist cameras:

```bash
ssh zed_box
cd repos/caip/ && source .venv/bin/activate && python stream_sender.py
```

Both must be up before the client starts. `start_receiving(timeout=10.0)` runs
during client init, so a stream that is down costs a 10 s wait and then an error
after the robot has already been connected.

A right-only policy consumes only **head + right wrist**, in the order the server
names in ping; `unscrew_v2` consumes all three. Start both wrist cameras either
way: `_get_raw_camera_images` fetches `LEFT_WRIST` and `RIGHT_WRIST`
unconditionally and asserts both shapes before the server's selection is
applied, so a dead left wrist stops a right-only run too.

With the GPU box now serving, finish the dry-run started in §D1 — still no
hardware, but this half needs the server:

```bash
python teleop/dryrun_relative_eef_deploy.py --server <gpu box ip> --port 5008
```

It pings, asserts `action_mode == relative_eef_rot6d`, sends one synthetic
bimanual observation, and checks the reply carries `arm_target_pose`
`(chunk, n_arms, 4, 4)` and `hand_target` `(chunk, 22*n_arms)` for the arms ping
declared — then IKs the first few rows. Shapes come from ping, so the same
command covers a right-only and a bimanual server without edits. A pass means
the wire contract and the IK convention agree before the robot is in the loop.

### D4. Client

```bash
cd teleop
python eval_genie.py --table-height <real table height> --no-save-data
```

`--server` defaults to `<ROBOT_IP>` and `--port` to 5008; pass them if the
GPU box is elsewhere. Two defaults that bite:

- **`--table-height` defaults to 0.85.** It feeds the reset planner's collision
  model, so a wrong value plans around a table that is not there.
- **recording defaults to ON.** The recording path deliberately writes the
  *pristine, unfilled* tactile, so a bring-up run lands in the dataset directory
  looking like a real episode. Keep `--no-save-data` until you are rolling out
  for real.

There is no task argument on the client. It takes the prompt from ping, verifies
the sha, and refuses a server started with `--task none`. It also compares its
vendored `teleop/vendor/tactile_health.py` against the server's — a refusal
there means the GPU box needs its `git pull`, not that something is broken.

### What to watch, in order

1. `Task: bowl  prompt_sha256=cb09decd1510` in the log, and the prompt echoed at
   the `Press [Enter]` gate. **Read it.** This is the only thing standing between
   you and a correctly-executed rollout of the wrong task.
2. Ping agreement: `arms=['right']` and two cameras in the order the server
   names — or `['left','right']` and three for `unscrew_v2`.
3. Right-only tasks: the left arm holds its constant target without drift. It is
   held at the FK of its default joint position, computed once at episode init.
   `unscrew_v2` drives both arms, so there is no held arm to watch.
4. One full chunk in free space before allowing the next.

## E. Blank-tactile injection, then contact

Inject blank tactile in software and confirm three things: the arm stops at the
next **chunk boundary** rather than mid-chunk, no stale rows are applied, and
recovery takes three consecutive live frames.

Everything under this is already covered offline — the filter through the real
30 Hz fetch loop on real episodes (gate 11), and the send-or-hold decision
itself (`test_request_gate.py`). So a failure here points at the robot
integration rather than at the logic, which is what makes it one confirmation
rather than a test campaign.

Then bowl contact, low speed.

### What a hold looks like

A hold is silence, not an error. The 300 Hz thread keeps commanding the last
target, so the arm simply stays where it is, and the client logs
`policy withheld (N consecutive): <reason>`. A chunk already executing is
deliberately allowed to finish — it was computed from healthy tactile, and
aborting mid-chunk would produce a deceleration profile nothing was validated
against. Worst case is a 1.8 s tail.

Nothing latches permanently. An over-window gap means the dropout outran what
training filled, not that the sensor is broken, so the hold clears by itself
after three consecutive strictly-live frames — a fill never counts as live,
which is what stops a flickering sensor from starting and stopping the arm.

## Chip

`chip_with_wrist` re-tests a checkpoint trained in July against a corpus
converted at the end of May, which is before most of the machinery this runbook
describes existed. It serves like `unscrew_v2` and passes F1 and F2, but three
of the guarantees the other tasks have are weaker here, and none of the three is
visible at launch unless you know to look.

**Its blank-tactile windows are a decision, not provenance.** Every other entry
in the registry copies its windows from that task's `run_convert.sh` branch, so
the online fill length provably equals the one its training data was built with.
This corpus predates those flags — there is no branch to copy. The windows are
the converter defaults (5 frames, all ten fingers), chosen to match
`unscrew_v2`. That is no more lenient than any policy this pipeline has used, so
it will not carry a dropout forward past what training contained; but a
blank-tactile hold on this task is unproven rather than validated, and section E
is a demonstration rather than a confirmation.

**The domain check cannot catch a wrong checkpoint.** `0530_pick_chip_no_wrist`
serves the same domain from entirely different weights. `action_type` and guard
2's `action_in_channels` (226 against its 240) both reject the pairing before
the model loads, so the mistake is caught — just not by the check that catches
it everywhere else.

**`arm_layout` is implicit.** The yaml predates the field, so `bimanual` comes
from the server default. The dry-run prints `(IMPLICIT default)` and the
registry entry asserts the expected layout, which is what makes relying on the
default acceptable rather than merely convenient. The deploy config was left a
byte-identical copy of the training config on purpose: being able to prove those
two are the same file is worth more than removing one default.

F3 and the byte-parity gate P have not been run for chip.

## Tong

Tong is served identically (`TASK=tong STEP=20000`) and passes every serving
gate, but its checkpoint has an unexplained problem: open-loop `mse_state` is
5.07 against bowl's 0.030 while `mse_action` matches bowl's, plus rot6d
non-orthonormality of 0.929 on the first chunk and a 112.9 deg rotation outlier.
A healthy action head beside a broken state echo is the signature of a
normalization problem rather than undertraining, so more steps will probably not
fix it. F1/F2/F3 cannot see this by construction — they verify that the server
faithfully transports and composes whatever the model emits, not that what it
emits is sane.

Resolve `state_eef` first, or demonstrate that the anchor and compose path
deployment actually uses is unaffected by it.
