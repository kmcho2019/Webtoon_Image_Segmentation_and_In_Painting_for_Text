import cv2
import numpy as np
import os
import math
import glob
import shutil
import argparse
import tempfile
#from diffusers import StableDiffusionInpaintPipeline
from diffusers import AutoPipelineForInpainting, LCMScheduler
from diffusers.utils import load_image, make_image_grid

from diffusers import AutoPipelineForInpainting
from PIL import Image, ImageDraw, ImageChops, ImageOps
import torch


# takes a black and white masks and expands region slightly to ensure that masks cover the text entirely
def mask_preprocessing(mask_dir):
    mask_list = list(glob.glob(os.path.join(mask_dir, '*.png')))

    for img_num in range(len(mask_list)):
        mask_path = mask_list[img_num]
        # Read Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Convert Mask to binary image
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        # Define structuring element for dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        # Apply Dilation to mask
        mask = cv2.dilate(mask, kernel, iterations = 3)
        # Convert dilated mask into binary image
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        # Save dilated mask back to original directory
        cv2.imwrite(mask_path, mask)

# A class that is able to process entire panel when the processing size is fixed
# Assumes that this is working on a long narrow panel and the process chunk size is smaller than width
# Processing function works on square grid of chunk_height x chunk_height
class DynamicOverlapImageProcessor:
    def __init__(self, processing_function, chunk_height=512, **kwargs):
        self.processing_function = processing_function
        self.chunk_height = chunk_height
        self.kwargs = kwargs

    def process_chunk(self, img_chunk, mask_chunk, original_img, start_y, end_y, last_row, start_x, end_x, last_col):
        if mask_chunk.getextrema()[1] == 0:  # All black mask
            return

        processed_chunk = self.processing_function(
            image=img_chunk,
            mask_image=mask_chunk,
            **self.kwargs
        )
        #print('img_chunk.size', img_chunk.size)
        #print('mask_chunk.size', mask_chunk.size)
        #print('processed_chunk.size', processed_chunk.size)
        img_chunk.save('debug_img.png')
        mask_chunk.save('debug_mask.png')
        processed_chunk.save('debug_processed_chunk.png')

        # Determine the portion to discard based on whether this chunk is the last row
        discard_rows = self.chunk_height - (end_y - start_y) if last_row else 0
        # Determine the portion to discard based on whether this chunk is the last col
        discard_cols = self.chunk_height - (end_x - start_x) if last_col else 0
        # Update original image
        cropped_processed_chunk = processed_chunk.crop((discard_cols, discard_rows, processed_chunk.width, processed_chunk.height))
        cropped_mask_chunk = mask_chunk.crop((discard_cols, discard_rows, mask_chunk.width, mask_chunk.height))
        cropped_original_chunk = img_chunk.crop((discard_cols, discard_rows, img_chunk.width, img_chunk.height))

        # Convert mask to a format suitable for blending (ensure it's single channel and 'L' mode)
        # Only the inpainted image corresponding to the white portion of mask is transferred back
        mask_for_blending = cropped_mask_chunk.convert('L')

        # Blend the processed chunk into the original image using the mask
        blended_chunk = Image.composite(cropped_processed_chunk, cropped_original_chunk, mask_for_blending)

        # Paste the blended chunk back into the original image
        original_img.paste(blended_chunk, (start_x, start_y))

        # Old Implementation where the entire chunk was copied back
        #original_img.paste(cropped_processed_chunk, (start_x, start_y))
        '''
        if last_row:
            # If this is the last row, only keep the unique, non-overlapping part of the chunk
            unique_rows = end_y - start_y - discard_rows
            cropped_processed_chunk = processed_chunk.crop((0, discard_rows, processed_chunk.width, discard_rows + unique_rows))
        else:
            # Otherwise, keep the entire chunk
            cropped_processed_chunk = processed_chunk.crop((0, discard_rows, processed_chunk.width, processed_chunk.height))

        original_img.paste(cropped_processed_chunk, (0, start_y + discard_rows))
        '''
        #cropped_processed_chunk = processed_chunk.crop((0, 0, processed_chunk.width, processed_chunk.height - discard_rows))
        #original_img.paste(cropped_processed_chunk, (0, start_y))

    def process_large_image(self, original_image, mask_image):
        original_image = original_image.convert("RGB")
        mask_image = mask_image.convert("L")

        width, height = original_image.size
        #print(width, height)
        # No need to pad as processing function is smaller than width
        '''
        # Pad the image horizontally
        pad_left = (1024 - width) // 2
        pad_right = 1024 - width - pad_left
        original_image = ImageOps.expand(original_image, (pad_left, 0, pad_right, 0), fill=0)
        mask_image = ImageOps.expand(mask_image, (pad_left, 0, pad_right, 0), fill=0)
        '''
        image_iter_num = 0
        # Loop through each vertical chunk with fixed height
        for start_y in range(0, height, self.chunk_height):
            for start_x in range(0, width, self.chunk_height):
                #print(original_image.size, mask_image.size)
                end_x = min(start_x + self.chunk_height, width)
                end_y = min(start_y + self.chunk_height, height)

                img_chunk = original_image.crop((start_x, start_y, end_x, end_y))
                mask_chunk = mask_image.crop((start_x, start_y, end_x, end_y))
                #print('start_x, end_x, start_y, end_y ',start_x, end_x, start_y, end_y)
                #print('image_iter_num, original_image.size, mask_image.size, img_chunk.size, mask_chunk.size')
                #print(image_iter_num, original_image.size, mask_image.size, img_chunk.size, mask_chunk.size)
                last_col = (end_x == width)
                last_row = (end_y == height)

                # Chunck need to be square
                # row: y, col: x
                if last_row and (not last_col):
                    img_chunk = original_image.crop((start_x, height-self.chunk_height, end_x, height))
                    mask_chunk = mask_image.crop((start_x, height-self.chunk_height, end_x, height))
                elif (not last_row) and last_col:
                    img_chunk = original_image.crop((width-self.chunk_height, start_y, width, end_y))
                    mask_chunk = mask_image.crop((width-self.chunk_height, start_y, width, end_y))
                elif last_row and last_col:
                    img_chunk = original_image.crop((width-self.chunk_height, height-self.chunk_height, width, height))
                    mask_chunk = mask_image.crop((width-self.chunk_height, height-self.chunk_height, width, height))
                #print(image_iter_num, original_image.size, mask_image.size, img_chunk.size, mask_chunk.size)
                #print('last_row, last_col', last_row, last_col)
                #print(image_iter_num)
                self.process_chunk(img_chunk, mask_chunk, original_image, start_y, end_y, last_row, start_x, end_x, last_col)
                #original_image.save(f'debug_intermediate_result_{image_iter_num:02}.png')
                image_iter_num = image_iter_num + 1

        '''
        # Remove padding to return to original dimensions
        original_image = original_image.crop((pad_left, 0, original_image.width - pad_right, original_image.height))
        '''
        return original_image

def diffusion_inpainting(pipe, prompt,negative_prompt, image, mask_image, generator, num_inference_steps=10, guidance_scale=4):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask_image,
        generator=generator,
        guidance_scale= guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]

    #the class works on numpy while the diffuser works on PIL
    return image


def get_args():
        parser = argparse.ArgumentParser(description='Batch Image Inpainting')
        parser.add_argument('--input_dir', default='./VLT_Ch116_test', help='Input image directory')
        parser.add_argument('--mask_dir', default = './mask_dir', help='Mask directory')
        parser.add_argument('--output_dir', default=None, help='Output directory')
        return parser.parse_args()


def main():
    print('main() started')
    args = get_args()
    print('argpase completed')
    # Configure directories based on command line arguments
    if args.output_dir is None:
        # Set directories
        args.output_dir = f'{os.path.basename(str(args.input_dir))}_output'

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)



    image_list = sorted(glob.glob(os.path.join(args.input_dir, '*.png')))
    mask_list = sorted(glob.glob(os.path.join(args.mask_dir, '*.png')))
    assert len(image_list) == len(mask_list), 'Number of images in input_dir and mask_dir is different'
    print('Configure Diffusion pipe')
    '''
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    '''
    '''
    # SDXL Pipe
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    '''
    # LCM Pipe
    pipe = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        variant="fp16", safety_checker=None
    ).to("cuda")
    # LCM: set scheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    # Load LcM-LoRA
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.fuse_lora()


    for i in range(len(image_list)):
        img_path = image_list[i]
        mask_path = mask_list[i]
        image = Image.open(img_path).convert('RGB')
        mask_image = Image.open(mask_path).convert('RGB')


        # Initialize and use processor
        processor = DynamicOverlapImageProcessor(
            processing_function=diffusion_inpainting,
            chunk_height=512,
            pipe=pipe,
            prompt='Seamlessly edited image, masterful photoshop job', #'nice, pristine, (((background)))',
            negative_prompt='text, error, cropped, out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature',
            num_inference_steps=10,
            generator= torch.manual_seed(0),
            guidance_scale=4,
        )
        output_image = processor.process_large_image(image, mask_image)
        # Save Image
        output_image.save(os.path.join(args.output_dir, os.path.basename(img_path)))


if __name__ == '__main__':
    print('Run Started')
    main()

