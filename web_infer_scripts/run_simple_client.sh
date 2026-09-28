#!/usr/bin/bash
# This file is from Genie-Envisioner (AgibotTech) at commit d54425c4, whose
# README licenses everything outside models/ltx_models, models/cosmos_models,
# models/pipeline and web_infer_utils/openpi_client under CC BY-NC-SA 4.0.
# It is redistributed here under those terms: see LICENSES/CC-BY-NC-SA-4.0.txt.
# NonCommercial use only; adaptations must carry the same licence.


IP_ADDRESS_OF_SERVER="localhost"

python3 web_infer_scripts/simple_client.py --host $IP_ADDRESS_OF_SERVER --port 8001 --env WM