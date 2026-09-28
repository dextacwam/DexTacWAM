# Modified by the DexTacWAM Authors, 2026.
# Originally from Genie-Envisioner (AgibotTech) at commit d54425c4.

import os
import sys

# torch.multiprocessing must be imported and configured BEFORE any other
# import that might transitively pull in `torch.utils.data.DataLoader`
# or spawn worker processes. Per-PyTorch docs (torch.multiprocessing.html
# #sharing-strategies) the strategy is process-global and is captured by
# DataLoader workers at fork time, so changing it after the first worker
# is spawned has no effect.
#
# Why we override the default ('file_descriptor'):
#   B1 production crashed at step 117/10000 (~6 min in) on our training host with
#   `OSError: [Errno 24] Too many open files` inside
#   torch/multiprocessing/reductions.py:619 (reduce_storage -> DupFd ->
#   resource_sharer.py). Root cause is the compounding of TWO sources of
#   per-worker FD pressure:
#     (1) DexVTAMDataset._mmap_cache (data/dex_vtam_dataset.py:783-823)
#         accumulates 4 mmap FDs (video / tactile / action / state)
#         per unique episode visited; never evicts. For the 488 corpus
#         this saturates at 488 * 4 = 1952 FDs / worker.
#     (2) The default 'file_descriptor' sharing strategy passes shared
#         tensors from each DataLoader worker back to the main process
#         by dup'ing a file descriptor per tensor. With num_workers=8
#         and prefetch_factor=2, dozens of FDs are in flight per worker
#         at any time. Each batch send burns an FD until the main
#         process unrefs the tensor.
#   Combined per-worker steady-state demand is ~2000+ FDs, well above
#   the default `ulimit -n 1024` on most Linux installs.
#
# 'file_system' strategy fix:
#   Switching to 'file_system' (the canonical PyTorch recipe for
#   `num_workers >> 1` + long-running training) shares tensors via
#   unique filenames in /tmp instead of FD-passing, cutting steady-state
#   per-worker IPC FD usage from ~20-100 to ~2-5. The filenames are
#   unlinked automatically when the tensor is freed (the receiver mmaps
#   the file then unlinks it; the underlying inode stays alive only
#   while at least one process holds it open), so /tmp does NOT grow
#   without bound. The only downside is a few microseconds of extra
#   IPC latency per tensor, invisible at our ~3 s/it scale.
#
# This is paired with a session-level `ulimit -n 65536` on that host (the
# launch tmux), giving us ~30x headroom over the worst-case combined
# FD demand (cache 1952 + IPC ~5 + baseline ~100 ~= 2057). The mmap
# cache itself remains unbounded -- a proper LRU eviction in
# DexVTAMDataset is deferred to before B2 (50k steps, larger model).
#
# Stage 1 trainer did not hit this because (a) its dataset is smaller
# (no _mmap_cache fast path; reads parquet on the fly) and (b) Stage 1
# runs are shorter (10-30k steps but with a much smaller corpus, so
# fewer unique episodes seen per worker).
import torch.multiprocessing as _mp
_mp.set_sharing_strategy('file_system')

import argparse
from utils import import_custom_class


def main():

    parser = argparse.ArgumentParser(
        description="Arguments for the main train program."
    )
    parser.add_argument('--config_file', type=str, required=True, help='Path for the config file')
    parser.add_argument('--runner_class_path', type=str, default="runner/ge_trainer.py")
    parser.add_argument('--runner_class', type=str, default="Trainer")
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to trained checkpoint, used in inference stage only')
    parser.add_argument('--n_validation', type=int, default=1, help='num of samples to predict, used in inference stage only')
    parser.add_argument('--n_chunk_action', type=int, default=1, help='num of action chunks to predict, used in action inference stage only')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save outputs, used in inference stage only')
    parser.add_argument('--domain_name', type=str, default="agibotworld", help='Domain name of the validation dataset, used in inference stage only')

    args = parser.parse_args()
    Runner = import_custom_class(
        args.runner_class, args.runner_class_path, 
    )
    

    if args.mode == "train":
        ### Trainer
        runner = Runner(args.config_file)
        runner.prepare_dataset()
        runner.prepare_models()
        runner.prepare_trainable_parameters()
        runner.prepare_optimizer()
        runner.prepare_for_training()
        runner.prepare_trackers()
        runner.train()

    elif args.mode == "infer":
        ### Inference
        runner = Runner(args.config_file, output_dir=args.output_path)
        if args.checkpoint_path is not None:
            runner.args.load_weights = True
            runner.args.load_diffusion_model_weights = True
            runner.args.diffusion_model['model_path'] = args.checkpoint_path
        runner.prepare_val_dataset()
        runner.prepare_models()
        runner.infer(
            n_chunk_action=args.n_chunk_action,
            n_validation=args.n_validation,
            domain_name=args.domain_name
        )

    else:
        raise NotImplementedError



if __name__ == "__main__":
  
    main()
    