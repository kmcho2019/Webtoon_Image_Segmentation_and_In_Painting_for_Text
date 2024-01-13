# Inpaint Subdirectory

## Overview
This subdirectory focuses on the inpainting component of the Webtoon Image Segmentation and In-Painting project. It leverages the Hugging Face Diffusers library, specifically employing the Stable Diffusion 1.5 LCM-LoRA inpaint mode for image processing.

## Key Features

### lcm_inpainting_script_command_line.py
- This script handles the inpainting process.
- It splits each panel into 512x512 chunks, processes each chunk, and then stitches them back together.
- Operates on a directory level, taking input, mask, and output directories as arguments.

### Usage
Run the script with the following command:
python lcm_inpainting_script_command_line.py --input_dir [Input Directory] --mask_dir [Mask Directory] --output_dir [Output Directory]

For detailed usage, you can refer to the help option:

python lcm_inpainting_script_command_line.py --help

or the code itself.

### Environment Setup
- A `huggingface_diffusers_environment.yaml` file is provided to help set up the necessary conda environment for running the script.

## Current Limitations and Future Work
- **Artifacts**: There are known issues with artifacts, especially on white backgrounds where strange objects may appear.
- **Future Improvements**: The next priority is to fine-tune the LCM LoRA model to better suit webtoon images and reduce artifacts.

## Contributing
Feel free to contribute to improving the inpainting process, especially in terms of model fine-tuning and artifact reduction. Your insights and pull requests are welcome!

## Acknowledgements
This inpainting solution is built using the Hugging Face Diffusers library and the Stable Diffusion 1.5 LCM-LoRA inpaint model. Special thanks to the developers and contributors of these tools.
