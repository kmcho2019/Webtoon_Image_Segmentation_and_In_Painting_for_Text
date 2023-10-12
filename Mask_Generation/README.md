This directory contains the mask generation/segmentation portion of the code.
Mask refers to the black and white image generated as the result of the code itself.
The segmentation model is based on MMDetection/mask2former implementation. 
The segmentation can be operated in batch with python mask_generation_script_command_line.py --input_dir input_dir --output_dir output_dir --score_thr 0.5 --model_weights iter_368750.pth --model_config custom_mask2former_config.py
The environment can be setup using the environment.yml folder using anaconda.
(Might need to update the prefix at the end of environment.yml. "prefix: /home/user_name/.conda/envs/new_env_name")
Python version is 3.9.16.
The weights were trained using dataset mixed with synthetic dataset.
There are two weights of iter_368750.pth and finetuned_iter_36875.pth, the latter had higher validation accuracy but anyone can be used.

