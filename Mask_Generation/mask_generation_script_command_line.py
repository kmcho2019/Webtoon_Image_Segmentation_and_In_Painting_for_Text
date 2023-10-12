import torch, torchvision
import mmdet
from mmcv.ops import get_compiling_cuda_version, get_compiler_version

import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector

import cv2
import numpy as np
import os
import math
import glob
import shutil
import argparse
import tempfile

# Code Taken from Webtoon_Batch_Processing_MMDetection_Instance_Segmentation_Mask2Former_Inference_Lama_InPainting.ipynb

# take vertical image(height > width) slice them up into squares
# conduct inference with model to each of them, also generate black and white
# mask with the inference results and stitch them up into original shape
def image_mask_generator(img_path, mask_dir, inference_dir, scratch_dir, model, score_thr = 0.1, show_inference_result = True):
  img_basename = os.path.basename(img_path)

  img = cv2.imread(img_path)
  height, width = img.shape[:2]

  img_num = int(math.ceil(height / width))

  # split vertical panel into square images
  for i in range(img_num):
    if i != img_num - 1:
      y_min = i * width
      y_max = (i+1) * width
      temp = img[y_min:y_max, :, :]
    else:
      temp = img[-width:, :, :]
    new_img_name = img_basename[:-4] + '_{:02n}'.format(i) + '.png'
    cv2.imwrite(os.path.join(scratch_dir, new_img_name), temp)
  # run inference on each of the images
  for i in range(img_num):
    img_name = img_basename[:-4] + '_{:02n}'.format(i) + '.png'
    img = mmcv.imread(os.path.join(scratch_dir, img_name))
    result = inference_detector(model, img)
    inference_file_name = img_name[:-4] + '_inference.png'
    if show_inference_result:
        show_result_pyplot(model, img, result, score_thr = 0.1)
    show_result_pyplot(model, img, result, score_thr = 0.1, out_file=os.path.join(scratch_dir, inference_file_name))

    bounding_box_prob_list = result[0][0]
    mask_list = result[1][0]
    object_num, img_row, img_col = np.shape(mask_list)
    square_mask_image = np.full((img_row, img_col), False)
    for object_index in range(object_num):
        object_prob = bounding_box_prob_list[object_index][-1:]
        if object_prob >= score_thr:
            square_mask_image = np.logical_or(square_mask_image, mask_list[object_index])
    # Convert Boolean to black and white image
    square_black_white_mask = (square_mask_image * 255).astype(np.uint8)
    # Save the binary image
    square_black_white_img_name = inference_file_name[:-4] + '_black_and_white_mask.png'
    cv2.imwrite(os.path.join(scratch_dir, square_black_white_img_name), square_black_white_mask)


  # stich the square images together
  for i in range(img_num):
    img_name = img_basename[:-4] + '_{:02n}'.format(i) + '.png'
    inference_file_name = img_name[:-4] + '_inference.png'
    square_black_white_img_name = inference_file_name[:-4] + '_black_and_white_mask.png'

    img = cv2.imread(os.path.join(scratch_dir, inference_file_name))
    mask_img = cv2.imread(os.path.join(scratch_dir, square_black_white_img_name))
    if i == 0: # intialize empty list at first
      concat_list = [img]
      mask_concat_list = [mask_img]
    elif i != img_num - 1: # append full square when not last
      concat_list.append(img)
      mask_concat_list.append(mask_img)
    else: # crop last image to preserve original shape
      last_square_height = height - (img_num - 1) * width
      concat_list.append(img[-last_square_height:, : ,:])
      mask_concat_list.append(mask_img[-last_square_height:, :, :])

  concat_img = cv2.vconcat(concat_list)
  mask_concat_img = cv2.vconcat(mask_concat_list)

  concat_img_name = img_basename[:-4] + '_inference_stitched.png'
  #   mask_concat_img_name = img_basename[:-4] + '_inference_black_and_white_mask_stitched.png'
  mask_concat_img_name = img_basename[:-4] + '_mask.png'
  cv2.imwrite(os.path.join(inference_dir, concat_img_name), concat_img)
  cv2.imwrite(os.path.join(mask_dir, mask_concat_img_name), mask_concat_img)

# takes a directory of images and generates mask for all of them using image_mask_generator
def batch_mask_generator(img_dir, mask_dir, inference_dir, scratch_dir, model, score_thr = 0.1, show_inference_result = False):
    image_list = glob.glob(os.path.join(img_dir, '*.png'))
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)
    else: # delete scratch_dir if it already exists then generate clean dir
        shutil.rmtree(scratch_dir)
        os.makedirs(scratch_dir)
    for img_path in image_list:
        image_mask_generator(img_path, mask_dir, inference_dir, scratch_dir, model, score_thr, show_inference_result)

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


def get_args():
        parser = argparse.ArgumentParser(description='Batch Image Segmentation')
        parser.add_argument('--input_dir', default='./VLT_Ch116_test', help='Input image directory')
        parser.add_argument('--output_dir', default=None, help='Output directory')
        parser.add_argument('--score_thr', type=float, default=0.5, help='Score threshold for segmentation')
        parser.add_argument('--model_weights', default='./iter_368750.pth', help='Path to model weights')
        parser.add_argument('--model_config', default='./custom_mask2former_config.py', help='Path to model config')
        parser.add_argument('--show_inference_result', default=0, help='Shows inference result while running, 0: Not Shown, 1: Shown')
        return parser.parse_args()

def main():
    print('main() started')
    args = get_args()
    print('argpase completed')
    # Configure directories based on command line arguments
    if args.output_dir is None:
        # Set directories
        args.output_dir = f'{os.path.basename(str(args.input_dir))}_output'
        mask_dir = os.path.join(args.output_dir, 'mask')
        inference_dir = os.path.join(args.output_dir, 'inference')
    else:
       mask_dir = args.output_dir
       inference_dir = args.output_dir + '_inference'

    if args.show_inference_result == 1:
        inference_show = True
    else:
        inference_show = False

    # Config and Weights based on MMDetection version 2.x does not work on version 3.x
    config_path = args.model_config #'./custom_mask2former_config.py'

    checkpoint_path = args.model_weights # './iter_30000.pth'

    # Set the device to be used for evaluation
    device='cuda:0'

    # Load the config
    config = mmcv.Config.fromfile(config_path)

    # Set pretrained to be None since we do not need pretrained model here
    # config.model.pretrained = None

    # Initialize the detector
    model = build_detector(config.model)

    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint_path, map_location=device)

    # Set the classes of models for inference
    model.CLASSES = checkpoint['meta']['CLASSES']

    # We need to set the model's cfg for inference
    model.cfg = config

    # Convert the model to GPU
    model.to(device)
    # Convert the model into evaluation mode
    model.eval()
    # Temporary directory for scratch files
    print('model setup completed')
    with tempfile.TemporaryDirectory() as scratch_dir:

        # Prepare output directories
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(inference_dir, exist_ok=True)

        # Batch processing
        print('batch processing started')
        batch_mask_generator(args.input_dir, mask_dir, inference_dir, scratch_dir, model, args.score_thr, show_inference_result=inference_show)

        # Post-processing of masks
        mask_preprocessing(mask_dir)

if __name__ == '__main__':
    print('Run Started')
    main()

'''
r'./VLT_Ch116_test' #r'/content/gdrive/MyDrive/Signals and Systems Research Project/VLT_Ch116_test'
dest_test_image_dir = r'./test_image_dir' + '_' + os.path.basename(src_test_image_dir) #r'/content/test_image_dir'
mask_dir = r'./mask_dir' + '_' + os.path.basename(src_test_image_dir)# r'/content/mask_dir'
inference_dir = r'./inference_dir' + '_' + os.path.basename(src_test_image_dir)#	r'/content/inference_dir'
scratch_dir = r'./scratch_dir' + '_' + os.path.basename(src_test_image_dir)#r'/content/scratch_dir'
processing_dir = r'./data_forPrediction' + '_' + os.path.basename(src_test_image_dir)#r'data_for_prediction'
output_dir = r'./output' + '_' + os.path.basename(src_test_image_dir)#r'/content/output'
score_thr = 0.1
show_inference_result = False
if os.path.exists(dest_test_image_dir):
    shutil.rmtree(dest_test_image_dir)
shutil.copytree(src_test_image_dir, dest_test_image_dir)
batch_mask_generator(dest_test_image_dir, mask_dir, inference_dir, scratch_dir, model, score_thr, show_inference_result)
mask_preprocessing(mask_dir)
'''
