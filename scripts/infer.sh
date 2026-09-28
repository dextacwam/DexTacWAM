#!/usr/bin/bash
# This file is from Genie-Envisioner (AgibotTech) at commit d54425c4, whose
# README licenses everything outside models/ltx_models, models/cosmos_models,
# models/pipeline and web_infer_utils/openpi_client under CC BY-NC-SA 4.0.
# It is redistributed here under those terms: see LICENSES/CC-BY-NC-SA-4.0.txt.
# NonCommercial use only; adaptations must carry the same licence.


script_path=${1}
echo $script_path

config_path=${2}
echo $config_path
ckp_path=${3}
output_path=${4}
domain_name=${5}

echo "Inference on 1 Nodes, 1 GPUs"
torchrun --nnodes=1 \
    --nproc_per_node=1 \
    --node_rank=0 \
    $script_path \
    --runner_class_path runner/ge_inferencer.py \
    --runner_class Inferencer \
    --config_file $config_path \
    --mode infer \
    --checkpoint_path $ckp_path \
    --output_path $output_path \
    --n_validation 1 \
    --n_chunk_action 10 \
    --domain_name $domain_name

