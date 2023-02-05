# Takes a directory of clean backgrounds and foregrounds(text cutouts) to generate a number of composed image and annotations for them
# Various augmentations are applied to the text such as linear scaling, rotation, transparency et cetera
# Also adds clean backgrounds with no foregrounds to provide a null case for the model
# Conversion to COCO json format is mostly taken from COCO_json_segmentation_dataset_generation_from_binary_mask.py



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

def DEBUG_view_cv2_image_array(image_arr):
    if image_arr.ndim == 2: # Black and white image
        plt.imshow(image_arr)
        plt.show()
    elif image_arr.ndim == 3: # Conventional color image
        if image_arr.shape[2] == 4: # alpha channel exists
            view = cv2.cvtColor(image_arr, cv2.COLOR_BGRA2RGBA)
        else:
            view = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
        plt.imshow(view)
        plt.show()
    else:
        print('Input shape is not correct')
        print(np.shape(image_arr))


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

# needed for contourIntersect as the function requires cv2 format contours
# works for only external contours
# input is a segmentation which stores contour in form of consecutive points of lists
def segmentation_list_2_cv2_contour_format(segmentation):
    tuple_input = [] # inserted to be converted to a tuple
    for connected_section in segmentation:
        points_for_section = []
        for i in range(len(connected_section) // 2):
            point = connected_section[2 * i:2 * i + 2]
            point = [point]
            points_for_section.append(point)
        tuple_input.append(np.array(points_for_section, dtype=np.int32))
    return tuple(tuple_input)

# check if two contours intersect or not
# run when is_there_no_bbox_collision() is triggered
# makes a much more detailed look at contours instead of just bounding boxes
# bounding box is checked first as it is less computationally intensive
# reference: https://stackoverflow.com/questions/55641425/check-if-two-contours-intersect
def contourIntersect(original_image_shape, contour1, contour2):
    if contour1 == [] or contour2 == []: #cannot intersect when one of the contours do not exist
        return False

    # contour1 = np.array(contour1).reshape((-1, 1, 2)).astype(np.int32)
    # contour2 = np.array(contour2).reshape((-1, 1, 2)).astype(np.int32)
    # Two separate contours trying to check intersection on
    contours = [contour1, contour2]
    contours = contour1 + contour2
    contours = segmentation_list_2_cv2_contour_format(contours)
    contour1_converted = segmentation_list_2_cv2_contour_format(contour1)
    contour2_converted = segmentation_list_2_cv2_contour_format(contour2)

    # print('contour1', contour1)
    # print('contour2', contour2)
    # print('contours', contours)
    # Create image filled with zeros the same size of original image
    # print(original_image_shape)
    blank = np.zeros(original_image_shape)

    # Copy each contour into its own image and fill it with '1'
    image1 = cv2.drawContours(blank.copy(), contour1_converted, -1, 1)
    image2 = cv2.drawContours(blank.copy(), contour2_converted, -1, 1)
    # DEBUG_view_cv2_image_array(image1)
    # DEBUG_view_cv2_image_array(image2)
    # Use the logical AND operation on the two images
    # Since the two images had bitwise and applied to it,
    # there should be a '1' or 'True' where there was intersection
    # and a '0' or 'False' where it didnt intersect
    intersection = np.logical_and(image1, image2)
    # DEBUG_view_cv2_image_array(intersection)
    # print(intersection.any())
    # Check if there was a '1' in the intersection
    return intersection.any()

# rotates foreground +/- max_rotation_angle
def apply_rotation(foreground, max_rotation_angle):
    rows, cols, channels = foreground.shape
    # Apply random rotation to foreground
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), random.uniform(-max_rotation_angle, max_rotation_angle), 1)
    rotated_foreground = cv2.warpAffine(foreground, rotation_matrix, (cols, rows))
    return rotated_foreground

# apply linear x and y scaling to foreground
# x and y scaling is applied separately and scaling lies between min_reduced_scale ~ (1 + max_expanded_scale)
def apply_linear_scaling(foreground, min_reduced_scale, max_expanded_scale):
    x_scale = random.uniform(min_reduced_scale, 1 + max_expanded_scale)
    y_scale = random.uniform(min_reduced_scale, 1 + max_expanded_scale)
    scaled_foreground = cv2.resize(foreground, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_CUBIC)
    return scaled_foreground

# apply sine and cosine scaling to x and y-axis
# randomly assigns wavelength and amplitude for both of the axis
# the wave length and amplitude range is given by max_wave_length and max_amplitude respectively
def apply_sine_cosine_distortion(foreground, max_wave_length=40, max_amplitude=5):
    x_wave_length = random.randint(1, max_wave_length)
    y_wave_length = random.randint(1, max_wave_length)
    x_amplitude = random.uniform(1,max_amplitude)
    y_amplitude = random.uniform(1,max_amplitude)

    # prevent outliers from distorting image too much
    # found that if amplitude large and wave length is small at the same time image more difficult to recognize
    # also found that it is fine if only one axis has high amplitude/wave_length ratio but problematic if both axis have high ratio
    # therefore make it so that if both axis have high ratio reduce limit ranges and run again
    if x_amplitude/x_wave_length > 3.1 and y_amplitude/y_wave_length > 3.1:
        x_wave_length = random.randint(5, max_wave_length)
        y_wave_length = random.randint(5, max_wave_length)
        x_amplitude = random.uniform(1, int(max_amplitude) // 2)
        y_amplitude = random.uniform(1, int(max_amplitude) // 2)

    temp_rows, temp_cols = foreground.shape[:2]
    # remap-1: form preliminary mapping array
    map_y, map_x = np.indices((temp_rows, temp_cols), dtype=np.float32)
    # remap-2: calculate distortion mapping using sine and cosine functions
    sin_x = map_x + x_amplitude * np.sin(map_y/x_wave_length)
    cos_y = map_y + y_amplitude * np.sin(map_x/y_wave_length)
    # remap-3: map image
    distorted_foreground = cv2.remap(foreground, sin_x, cos_y, cv2.INTER_LINEAR, None, cv2.BORDER_REPLICATE)
    return distorted_foreground

# draw a rotated line as kernel, then apply a convolution filter to an image with that kernel.
#refernce:https://stackoverflow.com/questions/40305933/how-to-add-motion-blur-to-numpy-array
#size - in pixels, size of motion blur
#angel - in degrees, direction of motion blur
# application_prob - probability of blur effect being applied
def apply_motion_blur(image, application_prob = 0.4):
    h, w, _ = np.shape(image)
    max_size = int(min(h, w) // 10) # max motion blur is set as 10 % of minimum axis size
    size_choice_array = [i for i in range(1, max_size + 1)] # need to add one to max_size to include it in choice array
    # if size is 1 then blur effect is not applied
    size_prob_array = [1-application_prob] + [round((application_prob / (len(size_choice_array)-1)), 4)] * (len(size_choice_array)-1)
    size_prob_array[0] = round(size_prob_array[0] + 1 - sum(size_prob_array), 4)
    # print(size_prob_array)
    # print(size_choice_array)
    # print(sum(size_prob_array))
    # size = random.randint(1, max_size)
    # select size within size_choice_array probability distribution based on size_prob_array
    size = np.random.choice(np.array(size_choice_array),None,True,p=np.array(size_prob_array))
    angle = random.uniform(-180, 180)
    k = np.zeros((size, size), dtype=np.float32)
    k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )
    k = k * ( 1.0 / np.sum(k) )
    return cv2.filter2D(image, -1, k)

# calculate brightness of image input image where the image has no alpha channel
# used to calculate brightness of background to compare against that of foreground in apply_brightness_calibration_and_randomize_color
def no_alpha_image_brightness(image):
    if len(image.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        # if (_, _, 4) == np.shape(image):
        #     print('Image with alpha transparency should not be used for this function')
        return np.average(np.linalg.norm(image, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(image)

# calculate brightness of image with alpha/transparency for a subsection of an image where the alpha value exceeds threshold
def alpha_image_geo_mean_brightness(image, threshold=127):
    # print(np.shape(image))
    h, w = image.shape[:2]
    mask = image[..., 3] >= threshold
    brightness = np.mean(image[..., :3][mask], axis=(0, 1))
    test = image[..., :3][mask]
    test = np.sum(test, axis = 1)
    test = test.astype(np.float)
    test = test/3
    # remove all zeros so that geometric mean can work
    test = test[test != 0]
    return np.exp(np.mean(np.log(test)))

# based on background image and image change brightness of foreground if there is too little difference
# also randomize color of image based on a probability
def apply_brightness_calibration_and_randomize_color(image, background_brightness, color_change_prob = 0.):

    if background_brightness < 40: # when background brightness is very dim make image brighter
        B_channel = random.randint(0, 255)
        G_channel = random.randint(0, 255)
        R_channel = random.randint(0, 255)

        # continue picking color when color dim as with background, pick again when luminosity is below 80
        # luminance formula source: https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color
        while (0.2126*R_channel + 0.7152*G_channel + 0.0722*B_channel) < 80:
            B_channel = random.randint(0, 255)
            G_channel = random.randint(0, 255)
            R_channel = random.randint(0, 255)
        image[:, :, 0] = B_channel
        image[:, :, 1] = G_channel
        image[:, :, 2] = R_channel
        mask = image[:, :, 3] > 128
        image[mask] = [B_channel, G_channel, R_channel, 255]
    else: # background is relatively bright
        if random.random() < color_change_prob: # when color change is triggered
            B_channel = random.randint(0, 255)
            G_channel = random.randint(0, 255)
            R_channel = random.randint(0, 255)
            # continue picking color when color is also bright as with background
            while math.sqrt((B_channel - 255) **2 + (G_channel - 255) ** 2 + (R_channel - 255) ** 2) < 40:
                B_channel = random.randint(0, 255)
                G_channel = random.randint(0, 255)
                R_channel = random.randint(0, 255)
            image[:,:,0] = B_channel
            image[:,:,1] = G_channel
            image[:,:,2] = R_channel
            mask = image[:,:,3] > 128
            image[mask] = [B_channel, G_channel, R_channel, 255]
    return image

# perform the image augmentation of foreground and also performs the overlaying process
# which is done using alpha blending process
# various augmentations include stretching, rotation, distortion, et cetera
def augment_and_overlay_images(foreground, background, rotation_angle=45, reduction_scale = 0.8, expansion_scale= 0.8, crop_limit = 0.4, min_foreground_opaqueness = 0.7):
    # Change color of image randomly
    color_changed_foreground = apply_brightness_calibration_and_randomize_color(image=foreground, background_brightness=no_alpha_image_brightness(background), color_change_prob=0.25)

    # Apply random rotation to foreground within (-rotation_angle, rotation_angle)
    rotated_foreground = apply_rotation(color_changed_foreground, rotation_angle)
    # Apply random scaling and distortion to the foreground
    scaled_foreground = apply_linear_scaling(rotated_foreground, reduction_scale, expansion_scale)
    # Apply non-linear remapping using trigonometry functions sine and cosine reference: https://bkshin.tistory.com/entry/OpenCV-15-%EB%A6%AC%EB%A7%A4%ED%95%91Remapping-%EC%98%A4%EB%AA%A9%EB%B3%BC%EB%A1%9D-%EB%A0%8C%EC%A6%88-%EC%99%9C%EA%B3%A1Lens-Distortion-%EB%B0%A9%EC%82%AC-%EC%99%9C%EA%B3%A1Radial-Distortion
    distorted_foreground = apply_sine_cosine_distortion(scaled_foreground,max_wave_length=40, max_amplitude=5)
    # DEBUG_view_cv2_image_array(distorted_foreground)
    # print('distorted_foreground shape',np.shape(distorted_foreground))
    motion_blur_foreground = apply_motion_blur(distorted_foreground, application_prob=0.4)
    # DEBUG_view_cv2_image_array(motion_blur_foreground)
    # print('If Identical Print False:', np.any(np.subtract(motion_blur_foreground, distorted_foreground))) # used to check if apply_motion_blur changes image when size is 1

    # augmented_foreground -> foreground after all augmentations have been applied

    augmented_foreground = motion_blur_foreground
    # Randomly assign a position to the foreground on the background
    fg_rows, fg_cols, fg_channels = augmented_foreground.shape
    bg_rows, bg_cols, bg_channels = background.shape
    x_offset = random.randint(int(-fg_cols * crop_limit), int(bg_cols - fg_cols * (1 - crop_limit)))
    y_offset = random.randint(int(-fg_rows * crop_limit), int(bg_rows - fg_rows * (1 - crop_limit)))
    inverse_x_offset = bg_cols - fg_cols - x_offset # offset measured from the opposite direction of x_offset (measured from right as opposed to being measured from left)
    inverse_y_offset = bg_rows - fg_rows - y_offset # offset measured from the opposite direction of y_offset (measured from bottom as opposed to being measured from top)
    # if both offset and inverse_offset are negative it means that scale foreground is larger than background problematic
    # repeat until it is fixed
    while (x_offset < 0 and inverse_x_offset < 0) or (y_offset < 0 and inverse_y_offset):
        crop_limit = 0.8 # more aggressive crop_limit to make it fit
        x_offset = random.randint(int(-fg_cols * crop_limit), int(bg_cols - fg_cols * (1 - crop_limit)))
        y_offset = random.randint(int(-fg_rows * crop_limit), int(bg_rows - fg_rows * (1 - crop_limit)))
        inverse_x_offset = bg_cols - fg_cols - x_offset  # offset measured from the opposite direction of x_offset (measured from right as opposed to being measured from left)
        inverse_y_offset = bg_rows - fg_rows - y_offset  # offset measured from the opposite direction of y_offset (measured from bottom as opposed to being measured from top)

    top_padding = y_offset if y_offset > 0 else 0
    bottom_padding = inverse_y_offset if inverse_y_offset > 0 else 0
    right_padding = inverse_x_offset if inverse_x_offset > 0 else 0
    left_padding = x_offset if x_offset > 0 else 0
    h_start_idx = abs(y_offset) if y_offset < 0 else 0
    h_end_idx = inverse_y_offset if inverse_y_offset < 0 else fg_rows
    w_start_idx = abs(x_offset) if x_offset < 0 else 0
    w_end_idx = inverse_x_offset if inverse_x_offset < 0 else fg_cols
    cropped_foreground = augmented_foreground[h_start_idx:h_end_idx, w_start_idx:w_end_idx, :]
    overlay_foreground = cv2.copyMakeBorder(cropped_foreground, top_padding, bottom_padding,left_padding,right_padding,cv2.BORDER_CONSTANT,None, value=0 )

    # draw contour based on alpha, but cv2.findContours need a binary image, binarize as resized foreground is not binary due to interpolation/resizing
    binarize_alpha = np.vectorize(lambda x: 0 if x < 100 else 255)
    # print('overlay_foreground shape', np.shape(overlay_foreground))
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
def augment_foreground(foreground, background, pre_existing_bboxes=None, pre_existing_contours = None, is_crowd = 0, image_id = 0, category_id = 1, annotation_id = 0, rotation_angle=45, reduction_scale = 0.8, expansion_scale= 0.8, crop_limit = 0.4, min_foreground_opaqueness = 0.7, attempts = 2):
    # Randomly rotate the foreground
    # if there is no pre_existing_bboxes set it to empty list
    if pre_existing_bboxes is None:
        pre_existing_bboxes = []
    if pre_existing_contours is None:
        pre_existing_contours = []
    # attempts variable refers to amount of tries program makes to accommodate colliding bounding boxes
    while attempts > 0:
        segmentations = []
        x_coordinate_list = []
        y_coordinate_list = []
        area = 0
        combined_image, faster_contours = augment_and_overlay_images(foreground, background, rotation_angle=rotation_angle, reduction_scale = reduction_scale, expansion_scale= expansion_scale, crop_limit = crop_limit, min_foreground_opaqueness = min_foreground_opaqueness)
        # DEBUG_view_cv2_image_array(cv2.drawContours(combined_image, faster_contours, -1, (255, 0, 0)))
        # print(faster_contours)
        for contour in faster_contours:
            # print('cv2.contourArea(contour)', cv2.contourArea(contour))
            # print(len(contour))
            if len(contour) > 2: # in terms of describing area contour with less than 3 points are not needed
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
        annotation = {
            'segmentation': segmentations,
            'iscrowd': is_crowd,
            'image_id': image_id,
            'category_id': category_id,
            'id': annotation_id,
            'bbox': bbox,
            'area': area
        }
        if is_there_no_bbox_collision(pre_existing_bboxes, bbox) is False:
            # print('BOUNDING BOX CLIPPING DETECTED, check more precisely with contours')
            # check again more precisely with contours

            if contourIntersect((combined_image.shape[0], combined_image.shape[1]), pre_existing_contours, segmentations) == True:
                attempts = attempts - 1 # try again with attempt after subtracting attempts
                # print('CONTOUR CLIPPING DETECTED RERUN, attempts remaining:', attempts)
                # DEBUG_view_cv2_image_array(combined_image)
            else:
                # no problem as contour themselves do not intersect
                # print('CONTOUR DOES NOT CLIP, accept current combined_image')
                return combined_image, annotation
        else:
            return combined_image, annotation
    # return original background and also return None in place for annotations
    # when number of attempts run out and there are still collisions
    return background, None

# uses augment_foreground() to generate composite images and their annotations
# returns coco_image_list and coco_annotation_list
def generate_image_and_coco_annotations(foreground_path, background_path, destination_path, max_foreground_num = 5, max_attempts_per_foreground = 3, num_images_per_background = 10, license_id = 1, category_id = 1, is_crowd= 0):
    if os.path.exists(destination_path):
        print('destination_path exists no need to create new directory')
    else:
        print('destination_path does not exist create new directory')
        os.makedirs(destination_path)

    foreground_list = os.listdir(foreground_path)
    background_list = os.listdir(background_path)
    foreground_list = glob.glob(os.path.join(foreground_path,'*.png'))
    background_list = glob.glob(os.path.join(background_path,'*.png'))

    total_generated_images = len(background_list) * num_images_per_background

    print(f'{len(background_list)} Background Images and {len(foreground_list)} Foreground Images Detected in Directory, Generating {total_generated_images} New Images Based on Multiplier {num_images_per_background} and {len(background_list)} Null Backgrounds')
    # license_id = 1
    image_id = 0
    annotation_id = 0
    # is_crowd = 0
    # category_id = 1  # 0 is for background and 1 is for text, as all annotation is text fix id to 1
    coco_category_list = []
    coco_image_list = []
    coco_annotation_list = []
    coco_license_list = []
    # first generate empty images(background with no foregrounds) to provide a null case for the model
    # can be useful in reducing false positives on an empty image
    print('Generating Empty Images With No Foregrounds to Provide Null Case For the Model')
    with tqdm.tqdm(total=len(background_list)) as pbar:
        for i in range(len(background_list)):
            current_background_path = background_list[i]
            current_background_filename = os.path.basename(current_background_path)
            new_background_filename = 'synth_null_background' + '_{:06d}_'.format(image_id) + current_background_filename
            new_background_path = os.path.join(destination_path, new_background_filename)
            shutil.copy(current_background_path, new_background_path) # copy background file without any modifications
            current_background_image = cv2.imread(current_background_path)
            (image_height, image_width, _) = np.shape(current_background_image)
            # Contain time elapsed since EPOCH in float
            ti_c = os.path.getctime(current_background_path)
            ti_m = os.path.getmtime(current_background_path)
            # Converting the time in seconds to a timestamp
            c_ti = time.ctime(ti_c)
            m_ti = time.ctime(ti_m)
            image_dict = {
                'id': image_id,
                'license': license_id,  # just set uniform license placeholder for now
                'file_name': new_background_filename,
                'height': image_height,
                'width': image_width,
                'date_captured': c_ti  # Use file creation date instead of last modification date
            }
            coco_image_list.append(image_dict)
            image_id = image_id + 1 # increment image_id after adding it to list
            pbar.update(1)
    print(f'Generating {total_generated_images} Composite Images With Maximum of {num_images_per_background} Foregrounds')
    time.sleep(0.1)
    with tqdm.tqdm(total=total_generated_images) as pbar:
        for background_num in range(len(background_list)):
            current_background_path = background_list[background_num]
            current_background_filename = os.path.basename(current_background_path)
            current_background_image = cv2.imread(current_background_path)
            for generation_num in range(num_images_per_background):
                new_composite_filename = 'synth_composite_image' + '_{:06d}'.format(image_id) + '_{:03d}_'.format(generation_num) + current_background_filename
                new_composite_path = os.path.join(destination_path, new_composite_filename)
                boxes = []
                contour_check_segmentation_accum_list = []

                for foreground_num in range(max_foreground_num):
                    current_foreground_path = random.choice(foreground_list) # pick random foreground from directory
                    current_foreground_image = cv2.imread(current_foreground_path, cv2.IMREAD_UNCHANGED)
                    # need to use cv2.IMREAD_UNCHANGED as foreground image also has alpha channel
                    # print('background_num, generation_num, foreground_num', background_num, generation_num, foreground_num)
                    if foreground_num == 0:
                        # for first pass use current_background_image as background
                        composite, current_annotation = augment_foreground(current_foreground_image,
                                                                           current_background_image,
                                                                           pre_existing_bboxes=boxes,
                                                                           pre_existing_contours=contour_check_segmentation_accum_list,
                                                                           is_crowd= is_crowd,
                                                                           image_id=image_id,
                                                                           category_id = category_id,
                                                                           annotation_id=annotation_id,
                                                                           attempts=max_attempts_per_foreground)
                    else:
                        # use composite image as background
                        composite, current_annotation = augment_foreground(current_foreground_image,
                                                                           composite,
                                                                           pre_existing_bboxes=boxes,
                                                                           pre_existing_contours=contour_check_segmentation_accum_list,
                                                                           is_crowd= is_crowd,
                                                                           image_id=image_id,
                                                                           category_id = category_id,
                                                                           annotation_id=annotation_id,
                                                                           attempts=max_attempts_per_foreground)
                    # print(generation_num, 'current composite shape', np.shape(composite))
                    if current_annotation == None:
                        # if current_annotation is None it means that no annotation was added
                        pass
                    else:
                        boxes.append(current_annotation['bbox'])
                        contour_check_segmentation_accum_list = contour_check_segmentation_accum_list + current_annotation['segmentation']
                        coco_annotation_list.append(current_annotation)
                        annotation_id = annotation_id + 1
                (composite_image_height, composite_image_width, _) = np.shape(composite)
                cv2.imwrite(new_composite_path, composite)
                # Contain time elapsed since EPOCH in float
                ti_c = os.path.getctime(new_composite_path)
                ti_m = os.path.getmtime(new_composite_path)
                # Converting the time in seconds to a timestamp
                c_ti = time.ctime(ti_c)
                m_ti = time.ctime(ti_m)
                image_dict = {
                    'id': image_id,
                    'license': license_id,  # just set uniform license placeholder for now
                    'file_name': new_composite_filename,
                    'height': composite_image_height,
                    'width': composite_image_width,
                    'date_captured': c_ti  # Use file creation date instead of last modification date
                }
                coco_image_list.append(image_dict)
                image_id = image_id + 1  # increment image_id after adding it to list
                pbar.update(1)
    return coco_image_list, coco_annotation_list



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
boxes = []
contour_check_segmentation_accum_list = []
overlay, annotation = augment_foreground(foreground, background)
print('annotation_segmentation',annotation['segmentation'])
print(len(annotation['segmentation']))
# DEBUG_view_cv2_image_array(overlay)
bbox = annotation['bbox']
contour_check_segmentation_accum_list = contour_check_segmentation_accum_list + annotation['segmentation']
boxes.append(bbox)
print(bbox)
print(boxes)
overlay, annotation = augment_foreground(foreground, overlay,pre_existing_bboxes=boxes, pre_existing_contours=contour_check_segmentation_accum_list)
# DEBUG_view_cv2_image_array(overlay)
if annotation == None:
    pass
else:
    boxes.append(annotation['bbox'])
    contour_check_segmentation_accum_list = contour_check_segmentation_accum_list + annotation['segmentation']
overlay, annotation = augment_foreground(foreground, overlay,pre_existing_bboxes=boxes, pre_existing_contours=contour_check_segmentation_accum_list)
# DEBUG_view_cv2_image_array(overlay)

cv2.imwrite('augment_foreground_output.png', overlay)

license_id = 1
category_id = 1
is_crowd = 0
coco_data_set_name = 'test_coco_dataset.json'
background_path = r'./test_background'
text_path = r'./test_sfx_images'
dest_path = r'./test_destination'

coco_image_list, coco_annotation_list = generate_image_and_coco_annotations(foreground_path=text_path,
                                                                            background_path=background_path,
                                                                            destination_path=dest_path,
                                                                            max_foreground_num=3,
                                                                            max_attempts_per_foreground=3,
                                                                            num_images_per_background=3,
                                                                            license_id=license_id,
                                                                            category_id=category_id,
                                                                            is_crowd=is_crowd)


current_datetime =  datetime.datetime.now()
coco_info = \
    {
        "year": datetime.date.today().year, # 2023
        "version": 1,
        "description": f'A synthetic COCO dataset generated using augment_foreground_test.py from background directory {background_path} and foreground(text) directory {text_path}',
        "contributor": 'kmcho2019', #GitHub id
        "url": r'https://github.com/kmcho2019/Webtoon_Image_Segmentation_and_In_Painting_for_Text',
        "date_created": str(current_datetime),
}


coco_category_list =[
    {
        "id": category_id,  # 1 normally can be changed
        "name": "text",
        "supercategory": "text"
    }
]

coco_license_list = [
    {
        "id": license_id,
        "name": 'PLACEHOLDER NOT DETERMINED YET',
        "url": 'PLACEHOLDER NOT DETERMINED YET'
    }
]
output_dataset =\
    {
        "info": coco_info, #info,
        "categories": coco_category_list, #[categories],
        "images": coco_image_list, #[image],
        "annotations": coco_annotation_list, #[annotation],
        "licenses": coco_license_list #[license]
}

# Write the JSON data to a file
with open(os.path.join(dest_path, coco_data_set_name), 'w') as json_file:
    json.dump(output_dataset, json_file)
