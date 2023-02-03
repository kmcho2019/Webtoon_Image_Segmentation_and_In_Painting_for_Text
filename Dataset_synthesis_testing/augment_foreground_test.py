import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import sys

# draw a rotated line as kernel, then apply a convolution filter to an image with that kernel.
#refernce:https://stackoverflow.com/questions/40305933/how-to-add-motion-blur-to-numpy-array
#size - in pixels, size of motion blur
#angel - in degrees, direction of motion blur
def apply_motion_blur(image, size, angle):
    k = np.zeros((size, size), dtype=np.float32)
    k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )
    k = k * ( 1.0 / np.sum(k) )
    return cv2.filter2D(image, -1, k)


# used to count the number of points in a contour
# was used to check between various methods of generating contours and their complexity
def count_num_points_in_contour(contours):
    total_count = 0
    for contour in contours:
        total_count = total_count + len(contour)
    return total_count

# check if there are any collisions between candidate_bbox and pre_existing_bboxes in image
# if there is no collision return True
# if there is collision return False
# a false result would mean that augment_foreground would not change image or give new annotations
def is_there_no_bbox_collision(pre_existing_bboxes:list, candidate_bbox:list)-> bool:
    if pre_existing_bboxes == []: # if pre_exiting_bboxes is empty as there are not preexisting bbox then there is can never be any collisions
        return True
    else: # check every bbox in pre_existing_bboxes with current_bbox for collisions
        [candidate_min_x, candidate_min_y, candidate_width, candidate_height] = candidate_bbox
        candidate_max_x = candidate_min_x + candidate_width
        candidate_max_y = candidate_min_y + candidate_height
        for current_bbox in pre_existing_bboxes:
            [current_min_x, current_min_y, current_width, current_height] = current_bbox
            current_max_x = current_min_x + current_width
            current_max_y = current_min_y + current_height
            if candidate_min_x < current_max_x and candidate_min_y < current_max_y and current_min_x < candidate_max_x and current_min_y < candidate_max_y:
                return False
        return True

# perform the image augmentation of foreground and also performs the overlaying process
# which is done using alpha blending process
# various augmentations include stretching, rotation, distortion, et cetera
def augment_and_overlay_images(foreground, background, rotation_angle=45, reduction_scale = 0.8, expansion_scale= 0.8, crop_limit = 0.4, min_foreground_opaqueness = 0.5):
    rows, cols, channels = foreground.shape
    # Apply random rotation to foreground
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), random.uniform(-rotation_angle, rotation_angle), 1)
    rotated_foreground = cv2.warpAffine(foreground, rotation_matrix, (cols, rows))

    # Apply random scaling and distortion to the foreground
    x_scale = random.uniform(reduction_scale, 1 + expansion_scale)
    y_scale = random.uniform(reduction_scale, 1 + expansion_scale)
    distorted_foreground = cv2.resize(rotated_foreground, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_CUBIC)
    # Apply non-linear remapping using trigonometry functions sine and cosine reference: https://bkshin.tistory.com/entry/OpenCV-15-%EB%A6%AC%EB%A7%A4%ED%95%91Remapping-%EC%98%A4%EB%AA%A9%EB%B3%BC%EB%A1%9D-%EB%A0%8C%EC%A6%88-%EC%99%9C%EA%B3%A1Lens-Distortion-%EB%B0%A9%EC%82%AC-%EC%99%9C%EA%B3%A1Radial-Distortion
    x_wave_length = random.randint(1, 20)
    y_wave_length = random.randint(1, 20)
    x_amplitude = random.uniform(1,5)
    y_amplitude = random.uniform(1,5)

    # prevent outliers from distorting image too much
    # found that if amplitude large and wave length is small at the same time image more difficult to recognize
    # also found that it is fine if only one axis has high amplitude/wave_length ratio but problematic if both axis have high ratio
    # therefore make it so that if both axis have high ratio reduce limit ranges and run again
    if x_amplitude/x_wave_length > 4 and y_amplitude/y_wave_length > 4:
        x_wave_length = random.randint(5, 20)
        y_wave_length = random.randint(5, 20)
        x_amplitude = random.uniform(1, 3)
        y_amplitude = random.uniform(1, 3)

    temp_rows, temp_cols = distorted_foreground.shape[:2]
    # remap-1: form preliminary mapping array
    map_y, map_x = np.indices((temp_rows, temp_cols), dtype=np.float32)
    # remap-2: calculate distortion mapping using sine and cosine functions
    sin_x = map_x + x_amplitude * np.sin(map_y/x_wave_length)
    cos_y = map_y + y_amplitude * np.sin(map_x/y_wave_length)
    # remap-3: map image
    distorted_foreground = cv2.remap(distorted_foreground, sin_x, cos_y, cv2.INTER_LINEAR, None, cv2.BORDER_REPLICATE)
    view = cv2.cvtColor(distorted_foreground, cv2.COLOR_BGR2RGBA)
    plt.imshow(view)
    plt.show()
    print('foreground shape', np.shape(foreground))
    print('distorted_foreground shape', np.shape(distorted_foreground))
    # Randomly assign a position to the foreground on the background
    fg_rows, fg_cols, fg_channels = distorted_foreground.shape
    bg_rows, bg_cols, bg_channels = background.shape
    x_offset = random.randint(int(-fg_cols * crop_limit), int(bg_cols - fg_cols * (1 - crop_limit)))
    y_offset = random.randint(int(-fg_rows * crop_limit), int(bg_rows - fg_rows * (1 - crop_limit)))
    inverse_x_offset = bg_cols - fg_cols - x_offset # offset measured from the opposite direction of x_offset (measured from right as opposed to being measured from left)
    inverse_y_offset = bg_rows - fg_rows - y_offset # offset measured from the opposite direction of y_offset (measured from bottom as opposed to being measured from top)
    top_padding = y_offset if y_offset > 0 else 0
    bottom_padding = inverse_y_offset if inverse_y_offset > 0 else 0
    right_padding = inverse_x_offset if inverse_x_offset > 0 else 0
    left_padding = x_offset if x_offset > 0 else 0
    h_start_idx = abs(y_offset) if y_offset < 0 else 0
    h_end_idx = inverse_y_offset if inverse_y_offset < 0 else fg_rows
    w_start_idx = abs(x_offset) if x_offset < 0 else 0
    w_end_idx = inverse_x_offset if inverse_x_offset < 0 else fg_cols
    cropped_foreground = distorted_foreground[h_start_idx:h_end_idx, w_start_idx:w_end_idx, :]
    overlay_foreground = cv2.copyMakeBorder(cropped_foreground, top_padding, bottom_padding,left_padding,right_padding,cv2.BORDER_CONSTANT,None, value=0 )

    # draw contour based on alpha, but cv2.findContours need a binary image, binarize as resized foreground is not binary due to interpolation/resizing
    binarize_alpha = np.vectorize(lambda x: 0 if x < 100 else 255)
    binary_contour_map_image = binarize_alpha(overlay_foreground[:,:,3])
    binary_contour_map_image = binary_contour_map_image.astype(np.uint8)
    faster_contours, _ = cv2.findContours(binary_contour_map_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    # overlayed[y_offset:y_offset + fg_rows, x_offset:x_offset + fg_cols] = distorted_foreground

    # set opaqueness of foreground between range
    foreground_opaqueness = random.uniform(min_foreground_opaqueness, 1.0)

    # add alpha channel to background to match that of overlay_foreground before cv2.addWeighted
    # alpha_background = np.dstack((background, np.full((bg_rows, bg_cols),255, dtype=np.uint8)))
    #use alpha blending technqiure to composite two images reference: https://learnopencv.com/alpha-blending-using-opencv-cpp-python/
    foreground_alpha = overlay_foreground[:,:,3]
    foreground_alpha = np.dstack([foreground_alpha] * 3) # make foreground_alpha into a three channel representation
    foreground_alpha = foreground_opaqueness * (foreground_alpha.astype(float) / 255)
    BGR_overlay_foreground = overlay_foreground[:,:,:-1] # overlay_foreground with alpha channel excluded
    alpha_weighted_foreground = cv2.multiply(foreground_alpha, BGR_overlay_foreground.astype(float))
    alpha_weighted_background = cv2.multiply(1 - foreground_alpha, background.astype(float))
    combined_image = cv2.add(alpha_weighted_foreground, alpha_weighted_background)
    combined_image = combined_image.astype(np.uint8)
    return combined_image, faster_contours


# rotation_angle maximum amount of angle that foreground is rotated
# distortion_scale image size scaled by 1 - distortion_scale ~ 1 + distortion_scale
# returns foreground background overlay image and annotation file corresponding to modified foreground
def augment_foreground(foreground, background, pre_existing_bboxes=None, is_crowd = 0, image_id = 0, category_id = 1, annotation_id = 0, rotation_angle=45, reduction_scale = 0.8, expansion_scale= 0.8, crop_limit = 0.4, min_foreground_opaqueness = 0.5, attempts = 2):
    # Randomly rotate the foreground
    # if there is no pre_existing_bboxes set it to empty list
    if pre_existing_bboxes is None:
        pre_existing_bboxes = []
    # attempts variable refers to amount of tries program makes to accommodate colliding bounding boxes
    while attempts > 0:
        segmentations = []
        x_coordinate_list = []
        y_coordinate_list = []
        area = 0
        combined_image, faster_contours = augment_and_overlay_images(foreground, background, rotation_angle=rotation_angle, reduction_scale = reduction_scale, expansion_scale= expansion_scale, crop_limit = crop_limit, min_foreground_opaqueness = min_foreground_opaqueness)
        for contour in faster_contours:
            area = area + cv2.contourArea(contour)
            partial_segmentation = contour.ravel().tolist()
            segmentations.append(partial_segmentation)
            x_coordinate_list = x_coordinate_list + partial_segmentation[0::2]
            y_coordinate_list = y_coordinate_list + partial_segmentation[1::2]
        min_x = min(x_coordinate_list)
        max_x = max(x_coordinate_list)
        min_y = min(y_coordinate_list)
        max_y = max(y_coordinate_list)
        width = max_x - min_x
        height = max_y - min_y
        bbox = [min_x, min_y, width, height]
        if is_there_no_bbox_collision(pre_existing_bboxes, bbox) is False:
            attempts = attempts - 1 # try again with attempt after subtracting attempts
        else:
            annotation = {
                'segmentation': segmentations,
                'iscrowd': is_crowd,
                'image_id': image_id,
                'category_id': category_id,
                'id': annotation_id,
                'bbox': bbox,
                'area': area
            }
            return combined_image, annotation
    # return original background and also return None in place for annotations
    # when number of attempts run out and there are still collisions
    return background, None



# distortion https://bkshin.tistory.com/entry/OpenCV-15-%EB%A6%AC%EB%A7%A4%ED%95%91Remapping-%EC%98%A4%EB%AA%A9%EB%B3%BC%EB%A1%9D-%EB%A0%8C%EC%A6%88-%EC%99%9C%EA%B3%A1Lens-Distortion-%EB%B0%A9%EC%82%AC-%EC%99%9C%EA%B3%A1Radial-Distortion
# np.set_printoptions(threshold=sys.maxsize)
# image = cv2.copyMakeBorder(src, top, bottom, left, right, borderType) # image padding function
# https://bkshin.tistory.com/entry/OpenCV-15-%EB%A6%AC%EB%A7%A4%ED%95%91Remapping-%EC%98%A4%EB%AA%A9%EB%B3%BC%EB%A1%9D-%EB%A0%8C%EC%A6%88-%EC%99%9C%EA%B3%A1Lens-Distortion-%EB%B0%A9%EC%82%AC-%EC%99%9C%EA%B3%A1Radial-Distortion
foreground = cv2.imread('image_bgra.png', cv2.IMREAD_UNCHANGED)
background = cv2.imread('195.png')
background = cv2.imread('131.png')
background = cv2.imread('159.png')

test = foreground[-40:,:,:]
print(np.shape(background))
print(np.shape(test))
background_height = background.shape[0]
background_width = background.shape[1]

test = cv2.copyMakeBorder(test, 10, 50,50,50,cv2.BORDER_CONSTANT,None, value=0 )
# plt.imshow(test)
# plt.show()
# cv2.imwrite('crop_test.png', test)

# plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
# plt.show()
# plt.imshow(background)
# plt.show()
overlay, contour_positions = augment_foreground(foreground, background)
cv2.imwrite('augment_foreground_output.png', overlay)