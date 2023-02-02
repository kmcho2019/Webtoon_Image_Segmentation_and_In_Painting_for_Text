import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import sys

def count_num_points_in_contour(contours):
    total_count = 0
    for contour in contours:
        total_count = total_count + len(contour)
    return total_count
# rotation_angle maximum amount of angle that foreground is rotated
# distortion_scale image size scaled by 1 - distortion_scale ~ 1 + distortion_scale
def augment_foreground(foreground, background, rotation_angle=45, reduction_scale = 0.8, expansion_scale= 0.8, crop_limit = 0.4, min_foreground_opaqueness = 0.5):
    # Randomly rotate the foreground
    rows, cols, channels = foreground.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), random.uniform(-rotation_angle, rotation_angle), 1)
    rotated_foreground = cv2.warpAffine(foreground, rotation_matrix, (cols, rows))

    # Apply random scaling and distortion to the foreground
    x_scale = random.uniform(reduction_scale, 1 + expansion_scale)
    y_scale = random.uniform(reduction_scale, 1 + expansion_scale)
    distorted_foreground = cv2.resize(rotated_foreground, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_CUBIC)
    # apply non-linear remapping using trigonometry functions sine and cosine reference: https://bkshin.tistory.com/entry/OpenCV-15-%EB%A6%AC%EB%A7%A4%ED%95%91Remapping-%EC%98%A4%EB%AA%A9%EB%B3%BC%EB%A1%9D-%EB%A0%8C%EC%A6%88-%EC%99%9C%EA%B3%A1Lens-Distortion-%EB%B0%A9%EC%82%AC-%EC%99%9C%EA%B3%A1Radial-Distortion
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
    print(int(-fg_rows * crop_limit), int(bg_rows - fg_rows * (1 - crop_limit)))
    overlayed = np.zeros_like(background)
    print(np.shape(overlayed))
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
    print(np.shape(background), np.shape(overlay_foreground))
    print(x_offset, y_offset)
    # plt.imshow(distorted_foreground[:,:,3])
    # plt.show()
    # draw contour based on alpha, but cv2.findContours need a binary image, binarize as resized foreground is not binary due to interpolation/resizing
    binarize_alpha = np.vectorize(lambda x: 0 if x < 100 else 255)
    binary_contour_map_image = binarize_alpha(overlay_foreground[:,:,3])
    binary_contour_map_image = binary_contour_map_image.astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary_contour_map_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#cv2.CHAIN_APPROX_TC89_KCOS)
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
    print(np.shape(foreground_alpha), np.shape(BGR_overlay_foreground))
    alpha_weighted_foreground = cv2.multiply(foreground_alpha, BGR_overlay_foreground.astype(float))
    alpha_weighted_background = cv2.multiply(1 - foreground_alpha, background.astype(float))
    combined_image = cv2.add(alpha_weighted_foreground, alpha_weighted_background)
    # combined_image = cv2.addWeighted(alpha_background, alpha_opaqueness, overlay_foreground, beta_opaqueness, 0)
    print(np.shape(combined_image))
    combined_image = combined_image.astype(np.uint8)
    view_combined_image = cv2.cvtColor(combined_image,cv2.COLOR_BGR2RGB)
    plt.imshow(view_combined_image)
    plt.show()

    #check contours by drawing them
    contour_image = cv2.drawContours(np.copy(combined_image), contours, -1, (0,0,255))
    view_combined_image = cv2.cvtColor(contour_image,cv2.COLOR_BGR2RGB)
    plt.imshow(view_combined_image)
    plt.show()
    simplified_contours = []
    for contour in contours:
        # found that for a larger image smaller epsilon is needed, so make epsilon inversely proportional to x/y_scale
        epsilon = (1/(x_scale * y_scale))* 0.005 * cv2.arcLength(contour, True) #0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) > 2:
            simplified_contours.append(approx)

    print('len(contours)', len(contours))
    print('contours size', count_num_points_in_contour(contours))
    print('len(simplified_contours)', len(simplified_contours))
    print('simplified_contours size', count_num_points_in_contour(simplified_contours))
    new_contour_image = cv2.drawContours(np.copy(combined_image), simplified_contours, -1, (0,255,0))
    new_view_combined_image = cv2.cvtColor(new_contour_image,cv2.COLOR_BGR2RGB)
    plt.imshow(new_view_combined_image)
    plt.show()
    # no simplification but just remove any len 1 or 2 contours
    truncated_contours = []
    for contour in contours:
        if len(contour) > 2:
            truncated_contours.append(contour)
    print('len(truncated_contours)', len(truncated_contours))
    print('truncated_contours size', count_num_points_in_contour(truncated_contours))

    truncated_contour_image = cv2.drawContours(np.copy(combined_image), simplified_contours, -1, (255,0,0))
    truncated_view_combined_image = cv2.cvtColor(truncated_contour_image,cv2.COLOR_BGR2RGB)
    plt.imshow(truncated_view_combined_image)
    plt.show()
    faster_contour_image = cv2.drawContours(np.copy(combined_image), faster_contours, -1, (255,255,0))
    faster_view_combined_image = cv2.cvtColor(faster_contour_image,cv2.COLOR_BGR2RGB)
    plt.imshow(faster_view_combined_image)
    plt.show()
    print('len(faster_contours)', len(faster_contours))
    print('faster_contours size', count_num_points_in_contour(faster_contours))
    # plt.imshow(cv2.cvtColor(combined_image,cv2.COLOR_BGR2RGB))
    # plt.show()
    # Get the outer contour positions of the augmented foreground on the background
    binary = cv2.cvtColor(overlayed, cv2.COLOR_BGR2GRAY)
    # _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    contour_positions = []
    for contour in contours:
        for position in contour:
            contour_positions.append([position[0][0] + x_offset, position[0][1] + y_offset])

    return overlayed, contour_positions
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