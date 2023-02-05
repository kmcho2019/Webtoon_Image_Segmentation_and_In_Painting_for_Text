# used to clean up noise in mask/Ground_Truth files
# many cases there are some isolated pixels surrounding the actual object itself due to inaccuracy with comparison data
import cv2
import numpy as np
import os
import tqdm

import glob


source_dir = r'.\combined_dataset_size_512\Ground_Truth_Morph_close_2_2'

mask_list = glob.glob(os.path.join(source_dir, '**', '*.png'), recursive=True)

for current_file_path in tqdm.tqdm(mask_list):
    # create diff file by comparing original and clean
    # print('i', i)
    # read mask in gray scale
    img = cv2.imread(current_file_path, cv2.IMREAD_GRAYSCALE)
    # form structuring kernel, square (4,4)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    # closing operation, closing chosen as the noise is mostly white on a black background
    # reference: https://bkshin.tistory.com/entry/OpenCV-19-%EB%AA%A8%ED%8F%B4%EB%A1%9C%EC%A7%80Morphology-%EC%97%B0%EC%82%B0-%EC%B9%A8%EC%8B%9D-%ED%8C%BD%EC%B0%BD-%EC%97%B4%EB%A6%BC-%EB%8B%AB%ED%9E%98-%EA%B7%B8%EB%A0%88%EB%94%94%EC%96%B8%ED%8A%B8-%ED%83%91%ED%96%87-%EB%B8%94%EB%9E%99%ED%96%87
    # closing = cv2.morphologyEx(img, cv2.MORPH_OPEN, k)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)

    # make sure that the image itself is binary
    _, binary = cv2.threshold(closing, 127, 255, cv2.THRESH_BINARY)
    # make sure that the format for storage is np.uint8
    binary = binary.astype(np.uint8)
    # save results back to same file
    cv2.imwrite(current_file_path, binary)

    #
