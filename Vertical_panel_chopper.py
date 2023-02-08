import os
import subprocess
import cv2
import numpy as np
import math
import glob
import datetime

import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import os
import tqdm
import shutil
import time
import datetime
import json
import math
import glob

RESIZE_WIDTH = 512

def make_check_dir(check_file_path):
    isExist = os.path.exists(check_file_path)
    if isExist == False:
        os.makedirs(check_file_path)

def image_processing_pipeline(f_original_image_name, f_resize_width=RESIZE_WIDTH):
    f_original_img = cv2.imread(f_original_image_name)

    np.shape(f_original_img)
    # print(np.shape(f_img))
    f_img_height = np.shape(f_original_img)[0]
    f_img_width = np.shape(f_original_img)[1]
    # print('img_height', f_img_height)
    # print('img_width', f_img_width)
    f_slice_num = math.ceil(f_img_height / f_img_width)  # number of vertical slices per image
    # print('slice_num', f_slice_num)
    f_resize_height = round((f_img_height / f_img_width) * f_resize_width)
    # print('resize_height', f_resize_height)

    f_resize_dim = (f_resize_width, f_resize_height)


    f_original_resize_img = cv2.resize(f_original_img, f_resize_dim,
                                     interpolation=cv2.INTER_AREA)  # cv2.INTER_CUBIC
    return f_original_resize_img, f_slice_num, f_resize_height

if __name__ == "__main__":
    source_dir = r'.\VLT_original_100_105'
    dest_dir = r'.\VLT_original_100_105_sliced_testing'

    source_dir = r'.\Dataset_synthesis_testing\background_collection\unprocessed_background_collection'
    dest_dir = r'.\Dataset_synthesis_testing\background_collection\processed_background_collection'

    make_check_dir(dest_dir)

    vertical_panel_list = glob.glob(os.path.join(source_dir, '**', '*.png'), recursive=True)
    series_dir_list = os.listdir(source_dir)
    for current_file_path in tqdm.tqdm(vertical_panel_list):
        # create diff file by comparing original and clean
        # print('i', i)
        current_file_dir = os.path.basename(os.path.dirname(current_file_path))
        current_file_name = os.path.basename(current_file_path)
        subtract_length = len(os.path.join(current_file_dir, current_file_name))
        series_dir_name = os.path.basename(current_file_path[:-(subtract_length + 1)]) # subtract chapter_dir and png name and also the \

        new_file_name = series_dir_name + '_' + current_file_dir + '_' + current_file_name[:-4] # remove the png tag

        # new_file_path = os.path.join(dest_dir, new_file_name)
        # create diff_slice file by processing slice
        # taken from Image_processing.ipynb



        # new processing pipeline
        original_resize_img, slice_num, resize_height = image_processing_pipeline(current_file_path, RESIZE_WIDTH)

        for j in range(slice_num):
            # print(j)
            if j == (slice_num - 1):
                original_crop_img = original_resize_img[(resize_height - RESIZE_WIDTH):, :]
            else:
                original_crop_img = original_resize_img[RESIZE_WIDTH * j: RESIZE_WIDTH * (j + 1), :]
            new_file_name_slice = new_file_name + '_slice_' + '{:03d}'.format(j + 1) + '.png'
            cv2.imwrite(os.path.join(dest_dir, new_file_name_slice), original_crop_img)
