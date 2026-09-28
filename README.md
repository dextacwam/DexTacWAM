<div align="center">

# [DexTacWAM: A Visuo-Tactile World-Action Model for Dexterous Manipulation](https://dextacwam.github.io/)

[Haoran Yuan](https://scholar.google.com/citations?user=PzdigUMAAAAJ)<sup>1,‡</sup> &nbsp;
[Zekai Wang](https://scholar.google.com/citations?user=Dngm3CYAAAAJ)<sup>2</sup> &nbsp;
[Boning Shao](https://scholar.google.com/citations?user=tlOWnSIAAAAJ)<sup>2</sup> &nbsp;
[Haoran Lu](https://luhr2003.github.io/)<sup>3</sup> &nbsp;
[Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/)<sup>2</sup> &nbsp;
[Ismini Lourentzou](https://isminoula.github.io/)<sup>1,†</sup> &nbsp;
[Wei Zhan](https://scholar.google.com/citations?user=xVN3UxYAAAAJ)<sup>2,†</sup>

<sup>1</sup>University of Illinois Urbana-Champaign &nbsp;
<sup>2</sup>University of California, Berkeley &nbsp;
<sup>3</sup>Northwestern University

<sup>‡</sup>Project lead &nbsp;&nbsp; <sup>†</sup>Equal advising, co-corresponding authors

[![Paper](https://img.shields.io/badge/arXiv-2609.24976-b31b1b.svg)](https://arxiv.org/abs/2609.24976)
[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb.svg)](https://dextacwam.github.io/)
[![License](https://img.shields.io/badge/License-Apache_2.0%20%2F%20CC_BY--NC--SA_4.0-green.svg)](LICENSE)

</div>

![DexTacWAM overview](assets/teaser.png)

## Overview

Dexterous manipulation depends on contact dynamics that are often only partially
observable from vision. Recent World-Action Models couple predictive video world
modeling with action generation, but remain largely vision-centric and therefore
cannot directly model these contact dynamics.

DexTacWAM is a visuo-tactile World-Action Model that encodes each fingertip
independently, aggregates the resulting features through a finger- and
pose-aware tactile compressor, and injects the tactile latent into a video
diffusion world model for joint visuo-tactile world modeling. Across six
contact-rich tasks on a 22-DoF bimanual platform it averages **70.6** against
**38.0** for the strongest baseline.

## Repository layout

```
models/tactile_models/   tactile encoder, compressor and projector
runner/                  three-stage trainers and the inferencer
data/                    datasets, caching and the layout contract
configs/                 the 16 configs behind the paper results
tests/                   unit and smoke tests
```

DexTacWAM builds on [Genie-Envisioner](https://github.com/AgibotTech/Genie-Envisioner-V1)
at commit `d54425c4`; that code is included here and the files we modified are
listed in [NOTICE](NOTICE).

## Installation

```bash
git clone https://github.com/dextacwam/DexTacWAM.git
cd DexTacWAM
pip install -r requirements.txt
```

Training was developed on 4x H200 NVL; stage 2 and 3 need multi-GPU, stage 1
fits on one.

Place the pretrained backbones where the configs expect them:

```
pretrained_models/ltx_video/
pretrained_models/genie_envisioner/GE_base_fast_v0.1.safetensors
```

## Checkpoints and data

| Component | Status |
| --- | --- |
| Tactile encoder and per-task policy checkpoints | October 2026 |
| Tactile interaction dataset (4 h, encoder adaptation) | October 2026 |
| Task demonstrations (~100 per task, six tasks) | October 2026 |

Until these land you can still run the full pipeline on your own LeRobot-format
corpus. Datasets go in `data/datasets_lerobot/<domain>/`, caches in
`data/cache/<name>/`, run outputs in `outputs/`.

## Data preparation

The normalization statistics for the six evaluation tasks are committed next to
their configs, so you only need to regenerate them if you bring your own corpus.
They must match the checkpoint you serve — different statistics silently
de-normalise actions wrong rather than failing:

```bash
python scripts/get_statistics.py \
    --relative \
    --arm-layout   bimanual \
    --data_root    data/datasets_lerobot/<domain>/data/chunk-000 \
    --data_name    <domain> \
    --data_type    eef \
    --action_key   action \
    --state_key    state \
    --save_path    configs/<task>/<domain>_relative_stats.json \
    --n-previous   4 \
    --action-chunk 54
```

`--data_type` must match the config's `action_space`, and `--arm-layout` must
name the layout the corpus was recorded with (`bimanual` or `right_only`); the
dataset hard-fails on a mismatch rather than silently mis-slicing. The file it
writes holds four blocks — `<domain>_eef`, `_delta_eef`, `_state_eef` and
`_relative_eef` — and the relative configs read the last two. `--n-previous` and
`--action-chunk` must equal the config's `n_previous` and `action_chunk` (4 and
54 for every released task).

`scripts/calculate_statistics.py` computes the same quantities for absolute
action spaces and has the outlier-filtering and gripper-clamping knobs; use it
only if you are not training on relative actions.

Then the offline cache. Training reads decoded frames from it rather than
re-decoding parquet every `__getitem__`, and the same cache serves both stage 2
and stage 3:

```bash
python scripts/preprocess_dex_vtam_cache.py \
    --data-root data/datasets_lerobot/<domain> \
    --domain <domain> \
    --cache-dir data/cache/<name> \
    --add-flow
```

## Training

The three stages run in order; each one's output is the next one's warm start.
After finishing a stage, point the next config at the run directory you just
produced — the paths committed here are from our runs and will not exist for you.

**Stage 1 — tactile encoder adaptation.** Single GPU. The pose and flow
statistics for the tactile interaction corpus
(`data/stats/wipe_plate_plus_wrap/`) ship with that dataset in October; the
`diverse_488` statistics every later stage needs are already in the repo.

```bash
python -m runner.tactile_vae_trainer --config configs/stage1_tactile_encoder.yaml
python -m runner.visual_vae_adapter_trainer --config configs/stage1_visual_vae_adapter.yaml
```

**Stage 2 — continual vision-to-touch learning.** Set `tactile_vae.model_path`
to the stage 1 checkpoint first. 80k steps, measured at 2.47 s/it on 4x H200 NVL
(~55 h); step 30000 is the stage 3 warm start.

```bash
torchrun --nnodes=1 --nproc_per_node=4 \
    main.py \
    --config_file configs/<task>/stage2_world_model.yaml \
    --runner_class_path runner/tactile_dit_trainer.py \
    --runner_class TactileDiTTrainer
```

**Stage 3 — action expert.** Set the world-model `model_path` to a stage 2
`step_*` directory. The action expert is randomly initialised
(`rand_init_action: true`); the world-model body is warm-started.

```bash
torchrun --nnodes=1 --nproc_per_node=4 \
    main.py \
    --config_file configs/<task>/stage3_action_expert.yaml \
    --runner_class_path runner/tactile_dit_trainer.py \
    --runner_class TactileDiTTrainer
```

`<task>` is one of `cube_place`, `cube_handover`, `wipe_whiteboard`, `tongs`,
`bowl_unstack`, `bottle_cap`. The visual-only world-model ablations live in
`configs/ablations/`.

Note that `torchrun` may report exit code 120 on an otherwise successful run
because of a cosmetic NCCL teardown race in accelerate + DeepSpeed. Treat the
presence of the final `step_*` checkpoint as the success signal, not the exit
code.

## Evaluation

Open-loop action evaluation against held-out episodes:

```bash
torchrun --nnodes=1 --nproc_per_node=1 \
    main.py \
    --config_file configs/<task>/stage3_action_expert.yaml \
    --runner_class_path runner/tactile_inferencer.py \
    --runner_class TactileInferencer \
    --mode infer \
    --checkpoint_path outputs/<stage3_run>/step_20000 \
    --output_path outputs/eval/<task> \
    --domain_name <domain> \
    --n_validation 10 \
    --n_chunk_action 30
```

`scripts/` also holds the analyses reported in the paper: world-model video
quality (`eval_wm_video_quality.py`), contact recall (`eval_wm_contact_recall.py`),
tactile flow visualisation (`eval_wm_tactile_flow_viz_offline.py`) and the
view-compression benchmark (`bench_tactile_view_compression.py`).

## Real-robot deployment

Deployment code lives on the **`deploy` branch**, not on `main`. It is a
server/client split: the policy server holds the world model and action expert
on the GPU workstation, and a thin client on the robot streams observations and
receives action chunks.

```bash
git checkout deploy
```

That branch adds `web_infer_scripts/` (policy server, robot client, offline
replay client and the server health-contract tests) plus the rollout runbook and
deployment gate checks under `docs/`. Keeping it separate means `main` stays
free of hardware-specific dependencies for the Sharpa fingertip sensors and the
Dexmate platform.

## Method

![Architecture](assets/architecture.png)

1. **Tactile-encoder adaptation.** A per-finger tactile encoder is adapted on
   four hours of tactile interaction data behind a frozen pretrained vision VAE.
2. **Continual vision-to-touch learning.** The pretrained video world model is
   extended to joint visuo-tactile prediction using roughly 100 demonstrations
   per task, without tactile midtraining of the video backbone, and retains
   visual prediction quality within 0.5 dB of its vision-only counterpart.
3. **Action expert training.** A randomly initialised action expert is trained
   on the same demonstrations, consuming predictive visuo-tactile features from
   a single world-model forward pass.

The finger- and pose-aware compressor maps ten fingertip streams to two
hand-level latents, retaining 89.4% of pre-fusion contact recall while enabling
2.26x faster training and 1.29x faster inference.

## Citation

```bibtex
@article{dextacwam2026,
  title   = {DexTacWAM: A Visuo-Tactile World-Action Model for Dexterous Manipulation},
  author  = {Yuan, Haoran and Wang, Zekai and Shao, Boning and Lu, Haoran and
             Darrell, Trevor and Lourentzou, Ismini and Zhan, Wei},
  journal = {arXiv preprint arXiv:2609.24976},
  year    = {2026}
}
```

## License

This repository is not under a single licence. Original DexTacWAM code, and
the Diffusers / LTX-Video / openpi code it builds on, are under the
[Apache License 2.0](LICENSES/Apache-2.0.txt).

Genie-Envisioner licenses everything outside `models/ltx_models`,
`models/pipeline` and `web_infer_utils/openpi_client`
under [CC BY-NC-SA 4.0](LICENSES/CC-BY-NC-SA-4.0.txt). Those files, and the
three DexTacWAM files substantially adapted from them, stay under that licence
because of its ShareAlike term, and **may not be used commercially**.

Every file states which licence applies in its own header, and [NOTICE](NOTICE)
maps it out.

## Contact

[lourent2@illinois.edu](mailto:lourent2@illinois.edu),
[wzhan@berkeley.edu](mailto:wzhan@berkeley.edu)
