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
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

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
diffusion world model for joint visuo-tactile world modeling.

Across six contact-rich dexterous manipulation tasks on a 22-DoF bimanual
platform, DexTacWAM achieves the highest score on every task, averaging **70.6**
versus **38.0** for the strongest baseline. Ablations attribute the gain to
modeling contact evolution as part of the predicted world state rather than to
tactile conditioning alone: removing tactile world modeling reduces the
four-task mean from **74.7** to **26.6** while keeping the same tactile features
and action expert.

## Release status

**Target: October 2026.** The code, checkpoints and data are being prepared for
public release, and this repository is where they will land. Watch or star it to
be notified.

| Component | Contents | Status |
| --- | --- | --- |
| Model and inference code | Visuo-tactile world model, tactile compressor, action expert | October 2026 |
| Pretrained checkpoints | Tactile encoder and per-task policies | October 2026 |
| Training code | Three-stage recipe: encoder adaptation, continual vision-to-touch learning, action expert | October 2026 |
| Tactile interaction dataset | 4 hours of multi-finger tactile interaction used for encoder adaptation | October 2026 |
| Task demonstrations | ~100 demonstrations per task across the six evaluation tasks | October 2026 |
| Hardware and deployment guide | 22-DoF bimanual platform, fingertip sensor integration, calibration | October 2026 |

Until then, the [project page](https://dextacwam.github.io/) hosts real-robot
rollouts, world-model predictions, tactile visualisations and additional
ablations.

## Method

![Architecture](assets/architecture.png)

Training proceeds in three stages:

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

Released under the [MIT License](LICENSE).

## Contact

Questions about the paper or the upcoming release are welcome by email:
[lourent2@illinois.edu](mailto:lourent2@illinois.edu),
[wzhan@berkeley.edu](mailto:wzhan@berkeley.edu).
