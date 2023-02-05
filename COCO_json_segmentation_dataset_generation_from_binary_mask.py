# Adopts some part of the code from Mask_R_CNN_image_preprocessing.py when it comes to isolating objects
# Converts binary mask into json annotation file in format of COCO segmentation
# Used to pre-proces image to isolate portion of mask into images
# Each black and white mask needs to be divided by objects and each objects needs a bounding box
# This program performs such operations and stores the characteristics of each object into json annotation file
# Uses xmeans from pyclustering to generate objects in a mask

# reference code: https://www.immersivelimit.com/create-coco-annotations-from-scratch
# Take black/white mask use clustering to group features, then color them
# Then modify reference code so that each object with different color can be stored as a separate feature


# Overall process
# Split images into train, valid, and test datasets
# Apply preprocessing(pyclustering xmeans) to groups features in each mask into objects
# Using this object information isolate

import os
import shutil

import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
# import math #used for math.isnan(i)

import pyclustering
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import read_image
from pyclustering.samples.definitions import IMAGE_SIMPLE_SAMPLES

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import tqdm
import time

import json #needed for COCO json file generation/manipulation.
#needed for creating polygons
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)

#needed to generate datetime
import datetime

# 18 distinct colors with black and white excluded
#taken from: https://sashamaps.net/docs/resources/20-colors/
# import COCO_json_segmentation_dataset_generation_from_binary_mask

group_colors = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128)]#, (255, 255, 255), (0, 0, 0)]

# source_path = r'.\coco_json_data_gen_test_data_source_20230125' #r'.\combined_dataset_checked_additonal_exclusions_1' #r'.\coco_json_data_gen_test_data_source_20230125'
# dest_path = '.\coco_json_data_gen_test_destination_20230125' #r'.\coco_json_combined_dataset_checked_additonal_exclusions_1' #r'.\coco_json_data_gen_test_destination_20230125'

source_path = r'.\combined_dataset_checked_additonal_exclusions_1' #r'.\coco_json_data_gen_test_data_source_20230125'
dest_path = r'.\coco_json_combined_dataset_checked_additonal_exclusions_1' #r'.\coco_json_data_gen_test_destination_20230125'

# source_path = r'.\coco_json_debug_20230128_source'
# dest_path = r'.\coco_json_debug_20230128_destination'


coco_data_set_name = dest_path[2:] + '_coco_dataset.json'
dest_subdir_name = ['Original', 'Colored_Ground_Truth'] #First should be the original images, the second should be the masks/ground_truth data with different colors

data_file_path = r'.\combined_dataset_checked_additonal_exclusions_1'#r'.\combined_dataset_checked'
new_processed_data_path = data_file_path + '_COCO_img_seg_dataset_format'
np.random.seed(42)
INIT_CENTERS = 2 #1
MAX_CENTERS = 18 # max capable by x-means is 20 but, only 18 is used because 18 colors are used to separate groups
MIN_DISTANCE = 10 #5 #3 #any distance greater than MIN_DISTANCE will result in separate objects
GEN_COLORED_MASK = True # Generates colored version of mask/Ground_Truth File at dest_subdir Colored_Ground_Truth when True

sub_directory_list = ['train_dir', 'valid_dir', 'test_dir']
sub_sub_directory_list = ['Ground_Truth', 'Original']
current_datetime = datetime.datetime.now()
## COCO data format
info =\
    {
        "year": int,
        "version": str,
        "description": str,
        "contributor": str,
        "url": str,
        "date_created": str #current_datetime,
}

image = \
    {
        "id": int,
        "width": int,
        "height": int,
        "file_name": str,
        "license": int,
        "flickr_url": str,
        "coco_url": str,
        "date_captured": str #current_datetime,
}

annotation = \
    {
        "id": int,
        "image_id": int,
        "category_id": int,
        "segmentation": list, #RLE or [polygon],
        "area": float,
        "bbox": list, #[x,y,width,height],
        "iscrowd": int #0 or 1,
}

license = \
    {
        "id": int,
        "name": str,
        "url": str,
}

categories = {
    "id": int,
    "name": str,
    "supercategory": str,
}


def make_check_dir(check_file_path):
    isExist = os.path.exists(check_file_path)
    if isExist == False:
        os.makedirs(check_file_path)

def load_data(path, split=0.1, remove_blank_masks = False):
    images = sorted(glob(os.path.join(path, "Original\\*")))
    masks = sorted(glob(os.path.join(path, "Ground_Truth\\*")))
    # images = sorted(glob(os.path.join(path, "images/*")))
    # masks = sorted(glob(os.path.join(path, "masks/*")))

    #Only remove blank masks when given the argument to do so (unlike Mask R CNN COCO can deal with blank masks)
    if remove_blank_masks:
        #assumes that images and masks have identical names
        #check for any completely dark masks and remove them (Mask R CNN require at least one object to be present)
        removed_files = []
        for i in range(len(masks)):
            img_mask = cv2.imread(masks[i])
            non_zero_exist = np.any(img_mask)
            if not non_zero_exist:
                removed_files.append(masks[i])
        print('removed_files')
        print(removed_files)
        #remove files in removed_files in both images and mask (assumed that they have identical names)
        # use os.path.basename to remove the path part of removed_files to simply extract the file name itself
        images = [x for x in images if os.path.basename(x) not in map(os.path.basename, removed_files)]
        masks = [x for x in masks if os.path.basename(x) not in map(os.path.basename, removed_files)]

        #alternative implementation of removing files (more data efficient)
        original_set = set(images)
        remove_set = set(map(os.path.basename, removed_files))
        filtered_set = original_set.difference(remove_set)
        images_filtered_list = list(filtered_set)

        original_set = set(masks)
        remove_set = set(map(os.path.basename, removed_files))
        filtered_set = original_set.difference(remove_set)
        masks_filtered_list = list(filtered_set)

        print('images', len(images))
        print('masks', len(masks))
        print('images_filtered_list', len(images_filtered_list))
        print('masks_filtered_list', len(masks_filtered_list))


    total_size = len(images)
    valid_size = int(split * total_size)
    test_size = int(split * total_size)
    print(total_size)
    # train_x, valid_x = train_test_split(images, test_size=split, train_size= (1-split), random_state=42)
    # train_y, valid_y = train_test_split(masks, test_size=split,train_size= (1-split), random_state=42)

    # train_x, test_x = train_test_split(train_x, test_size=split,train_size= (1-split), random_state=42)
    # train_y, test_y = train_test_split(train_y, test_size=split, train_size= (1-split),random_state=42)
    train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

#given img np array generate dictionary that matches non-zero value to id and coordinate also generate group_data for clustering algorithm
def data_and_id_coord_dict_generator(img):
  data = []
  counter = 0
  cluster_id_coord_dict = {}
  for i in range(np.shape(img)[0]):
    for j in range(np.shape(img)[1]):
      # img_copy[i,j] = img[i, j, 0]
      if img[i, j, 0] > 0. or img[i, j, 1] > 0. or img[i, j, 2] > 0.:
        data.append([j, i])
        cluster_id_coord_dict[counter] = (i,j) # id: (row_coord, col_coord)
        counter = counter + 1
  return data, cluster_id_coord_dict

#given imp np array generate initiial xmeans clusters (ultimately used to separate objects and generate bounding boxes)
def run_xmeans_on_img(img, initial_centers=INIT_CENTERS, max_centers=MAX_CENTERS):
  amount_initial_centers = initial_centers
  max_cluster_num = max_centers
  data, cluster_id_coord_dict = data_and_id_coord_dict_generator(img)
  # Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will start analysis.
  initial_centers = kmeans_plusplus_initializer(data, amount_initial_centers).initialize()

  # Create instance of X-Means algorithm. The algorithm will start analysis from 1 clusters, the maximum number of clusters that can be allocated is 20.
  x_means_instance = xmeans(data, initial_centers, max_cluster_num)
  x_means_instance.process()

  # Extract clustering results: clusters and their centers
  clusters = x_means_instance.get_clusters()
  centers = x_means_instance.get_centers()
  return clusters


#chat gpt code

#calculate min_distance between two groups using id_coord_dict
def group_min_distance(img, group_1, group_2, id_coord_dict):
  min_distance = np.shape(img)[0]
  for id1 in group_1:
    coords1 = id_coord_dict[id1]
    for id2 in group_2:
        coords2 = id_coord_dict[id2]
        distance = np.sqrt((coords1[0] - coords2[0])**2 + (coords1[1] - coords2[1])**2)
        if min_distance > distance:
          min_distance = distance
  return min_distance

#take n_components list into index groups
#ex: [0 0 1 2 3 4 4 4 4 5] => [[0, 1], [2], [3], [4], [5, 6, 7, 8], [9]]
def group_indexes(lst):
    # Create a dictionary to store the groups
    groups = {}
    # Iterate through the list
    for i, value in enumerate(lst):
        # Check if the value is in the dictionary
        if value in groups:
            # If it is, add the index to the corresponding group
            groups[value].append(i)
        else:
            # If it is not, create a new group for the value
            groups[value] = [i]
    # Convert the dictionary to a list of lists
    groups = [groups[key] for key in groups]
    return groups

def merge_groups_graph_reduce_WIP(img, groups, id_coor_dict, min_distance=MIN_DISTANCE):
  if len(groups) == 1: # no need to merge further when number of groups is already one
    return groups
  if len(groups) == 0:
    print('NUMBER OF GROUPS 0, CHECK CODE')
    print(groups)
    return groups
  #lambda function that is used to generate connectivity graph(only convert distance of 0<d<=min_distance to 1, otherwise 0)
  connectivity_converter = lambda t, min_distance: 0 if t == 0. else (1 if t <= min_distance else 0)
  connectivity_matrix_converter = np.vectorize(connectivity_converter)

  distance_matrix = np.zeros((len(groups), len(groups)))
  # Initialize the merged groups list
  merged_groups = []
  for i in range(0,len(groups)-1):
    for j in range(i+1, len(groups)):
      distance_matrix[i,j] = group_min_distance(img, groups[i], groups[j], id_coor_dict)
      distance_matrix[j,i] = distance_matrix[i,j]
  # print('distance_matrix shape:', np.shape(distance_matrix))
  # print('distance_matrix:', distance_matrix)
  minval = np.min(distance_matrix[np.nonzero(distance_matrix)])
  if minval > min_distance: #np.all(distance_matrix > min_distance): #No more merging is needed
    return groups #no more merging so return as it is
  else:
    #generate graph matrix using distance_matrix, abstracts groups into nodes and edges only exist if groups are within min_distance
    #connected nodes(components) can be joined together to be eliminated at once
    connect_mat = connectivity_matrix_converter(distance_matrix, min_distance)
    graph = csr_matrix(connect_mat)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    indexed_labels = group_indexes(labels)
    non_single_indexed_labels = [group for group in indexed_labels if len(group) > 1] #remove any single variable lists
    single_indexed_labels = [group for group in indexed_labels if len(group) == 1] #collection of lists with single entries
    # print('labels', labels)
    # print('indexed_labels',indexed_labels)
    # print('non_single_indexed_labels',non_single_indexed_labels)
    # print('single_indexed_labels', single_indexed_labels)

    for i in [item for group in single_indexed_labels for item in group]: #flatten the single_indexed_labels
        merged_groups.append(groups[i])
    for sub_list in non_single_indexed_labels:
      merge_index_list = []
      for j in sub_list:
        merge_index_list = merge_index_list + groups[j]
      merged_groups.append(merge_index_list)
    #Prevent infinite recursion by comparing groups, if same (or have same number of groups) return merged_groups and stop recursion
    #Added as some examples tended to take dozens of minutes to process
    is_merged_groups_and_groups_same = False
    element_sort_groups = [sorted(x) for x in groups]
    element_sort_merged_groups = [sorted(x) for x in merged_groups]
    sub_list_sort_groups = sorted(element_sort_groups)
    sub_list_sort_merged_groups = sorted(element_sort_merged_groups)
    if sub_list_sort_groups == sub_list_sort_merged_groups or len(sub_list_sort_groups) == len(sub_list_sort_merged_groups):
        is_merged_groups_and_groups_same = True
    if is_merged_groups_and_groups_same: #no point in continuing recursion when operation does not change result or if group number is the same
        return merged_groups
    return merge_groups_graph_reduce_WIP(img, merged_groups, id_coor_dict, min_distance)

#visulize merged groups generated by merge_groups_graph_reduce_WIP()
#each group has different color schemes
def visualize_groups(img, groups, coord_dict):
    # Create a copy of the image to avoid modifying the original
    visualized_img = img.copy()
    # Generate a random color for each group
    colors = [(random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)) for _ in range(len(groups))]
    # Iterate through all groups
    for group, color in zip(groups, colors):
        # Color all values in the group
        for id in group:
          # print(id)
          # print(np.shape(visualized_img[visualized_img == id] ))
          # print(visualized_img[visualized_img == id] )
          coordinate = coord_dict[id]
          row = int(coordinate[0])
          col = int(coordinate[1])
          # print(row, col)
          # print(np.shape(visualized_img))
          # print(color)
          visualized_img[row, col] = color
          # visualized_img[visualized_img == id] = color
    # Display the visualized image
    plt.imshow(visualized_img)
    plt.show()

#get bounding box of a group(input in form of lists of ids)
#top left coordinate and bottom right coordinate
def get_bounding_box(ids, id_to_coords):
    # Initialize the bounding box coordinates
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    # Iterate through the IDs
    for id in ids:
        # Get the coordinates for the current ID
        # x, y = id_to_coords[id]
        y, x = id_to_coords[id] #dictionary storage format is (row, col), which corresponds to y, x
        # Update the bounding box coordinates if necessary
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    # Return the bounding box coordinates as a tuple
    return (min_x, min_y), (max_x, max_y)

#draws bounding box given groups and id_coord_dictionary using get_bounding_box to generate the boxes
def draw_bounding_box(img, groups_ids, id_to_coords):
    # Create a copy of the image
    img_copy = img.copy()
    # Get the bounding box coordinates
    print(groups_ids)
    for ids in groups_ids:
      print(ids)
      bbox = get_bounding_box(ids, id_to_coords)

      # color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
      # Draw the bounding box on the image
      cv2.rectangle(img_copy, bbox[0], bbox[1], (0, 255, 0), 2)
    # Display the image with the bounding box
    plt.imshow(img_copy)
    plt.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Use Xmeans clustering to find num_instances (number of objects)
def extract_bboxes(filename):
    img = cv2.imread(filename)
    (h, w, _) = np.shape(img)
    _,cluster_id_coord_dict = data_and_id_coord_dict_generator(img) #first output(data) ignored in this case as data is only needed inside run_xmeans_on_img() function
    xmean_clusters = run_xmeans_on_img(img)
    merged_groups = merge_groups_graph_reduce_WIP(img, xmean_clusters, cluster_id_coord_dict)
    boxes = []
    box_masks = []
    for group_num in range(len(merged_groups)):
      (min_x, min_y), (max_x, max_y) = get_bounding_box(merged_groups[group_num], cluster_id_coord_dict)
      object_mask = np.zeros((max_y - min_y + 1, max_x - min_x + 1))
      boxes.append([min_x, min_y, max_x, max_y])
      object_mask = img[min_y: max_y + 1, min_x: max_x + 1, 0] #within bounding box region extract only R channel (should be without problems as RGB channel should be identical)
      object_mask[object_mask != 0] = 1 # set non-zero values to one
      box_masks.append(object_mask)
    return box_masks, boxes, w, h

# Take objects and their bounding boxes extracted from extract_bboxes() to give a colored version of the mask
# input_masks consist of list of 2D binary arrays where 1 corresponds to object, 0 corresponds to background
# input_boxes consist of list of lists with each list being [min_x, min_y, max_x, max_y] of each object
# input_masks, input_boxes are taken from the outputs of extract_bboxes
def colored_mask_generator(original_mask: np.array, input_masks: list, input_boxes: list):
    colored_mask = original_mask.copy() #original mask is a numpy array from cv2, copy original so that it is not affected
    number_of_objects = len(input_boxes)
    for i in range(number_of_objects):
        empty_object_img = np.zeros((original_mask.shape[0], original_mask.shape[1], 3), dtype=np.uint8)
        [min_x, min_y, max_x, max_y] = input_boxes[i]
        binary_image = input_masks[i]
        channel_expanded_input_mask = np.repeat(binary_image[:, :, np.newaxis], 3, axis=2)
        empty_object_img[min_y:max_y + 1, min_x:max_x + 1] = channel_expanded_input_mask

        # Get the pixels within the boundary that are white
        mask = np.all(empty_object_img[min_y:max_y + 1, min_x:max_x + 1] == [1, 1, 1], axis=-1)
        # Change the value of the white pixels to the specified color
        colored_mask[min_y:max_y + 1, min_x:max_x + 1][mask] = group_colors[i]
    return colored_mask

# Takes object mask taken from extract_bboxes() to create annotation for coco json
# Reference Code: https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
def create_annotation_from_partial_mask(partial_mask, bounding_box, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each partial_mask
    # Sometimes there can be multiple contours for objects as text is not always connected.

    [mask_min_x, mask_min_y, mask_max_x, mask_max_y] = bounding_box # relative position of partial_mask in respect of entire image


    #### Original Implementation Start, not from reference website:
    (rows, cols) = np.shape(partial_mask)
    padded_mask = np.zeros((rows + 2, cols + 2), dtype=np.uint8) # np.uint8 as it caused errors in cv2.findContours
    # print('np.shape(partial_mask)', np.shape(partial_mask))
    # print('np.shape(padded_mask)', np.shape(padded_mask))
    # print('rows + 2, cols + 2', rows + 2, cols + 2)

    padded_mask[1:rows + 1, 1:cols + 1] = partial_mask

    # plt.imshow(padded_mask)
    # plt.show()
    cv2_contours, _ = cv2.findContours(padded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    points = []
    x_coordinate_list = []
    y_coordinate_list = []
    area = 0
    for contour in cv2_contours:
        # epsilon = 0.01 * cv2.arcLength(contour, True)
        # approx = cv2.approxPolyDP(contour, epsilon, True)
        # object_area = object_area + cv2.contourArea(approx)
        # restored_approx = np.subtract(approx, 1) # subtract all coordinates x and y by one to account for padding
        # partial_points = restored_approx.ravel().tolist()
        if len(contour) >= 3: # only add only if contour form a discernable area
            area = area + cv2.contourArea(contour)
            recalibrated_contour = np.subtract(contour, 1) # subtract all coordinates x and y by one to account for padding
            partial_points = recalibrated_contour.ravel().tolist()
            #move the coordinates to the entire object's position in image
            # needed as the current coordinates are based on just the object, not the entire image
            x_coordinate_list = x_coordinate_list + partial_points[0::2]
            y_coordinate_list = x_coordinate_list + partial_points[1::2]
            # conversion to coco json segmentation format
            saved_seg_points = [mask_min_x + x if i % 2 == 0 else mask_min_y + x for i, x in enumerate(partial_points)]

            points.append(saved_seg_points)



    # print(points)

    cv2_min_x = min(x_coordinate_list)
    cv2_max_x = max(x_coordinate_list)
    cv2_min_y = min(y_coordinate_list)
    cv2_max_y = max(y_coordinate_list)
    cv2_width = cv2_max_x - cv2_min_x
    cv2_height = cv2_max_y - cv2_min_y
    # print('cv2_points\n', points)
    # print('\ncv2_min_x, cv2_min_y, cv2_max_x, cv2_max_y',cv2_min_x, cv2_min_y, cv2_max_x, cv2_max_y)
    # print('\ncv2_area', object_area)
    cv2_contour_point_num = 0
    for sub_list in points:
        cv2_contour_point_num = cv2_contour_point_num + len(sub_list)
    # print('cv2_contour_point_num', cv2_contour_point_num)
    segmentations = points
    bbox = [cv2_min_x + mask_min_x, cv2_min_y + mask_min_y, cv2_width, cv2_height]

    #### Original Implementation End
    '''
    
    #add zero padding to partial_mask as contours cannot handle cases where mask touches edge
    (rows, cols) = np.shape(partial_mask)
    padded_mask = np.zeros((rows + 2, cols + 2), dtype=np.uint8) # np.uint8 as it caused errors in cv2.findContours
    # print('np.shape(partial_mask)', np.shape(partial_mask))
    # print('np.shape(padded_mask)', np.shape(padded_mask))
    # print('rows + 2, cols + 2', rows + 2, cols + 2)

    padded_mask[1:rows + 1, 1:cols + 1] = partial_mask
    # print('partial_mask\n', partial_mask)
    # print('padded_mask\n', padded_mask)

    # print(padded_mask)

    contours = measure.find_contours(padded_mask, 0.5, positive_orientation='low')
    # print('\ncontours\n', len(contours))
    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            # move the coordinates to the entire object's position in image
            # needed as the current coordinates are based on just the object, not the entire image
            contour[i] = (col - 1 + mask_min_x, row - 1 + mask_min_y)


        #Make polygon and simplify
        poly = Polygon(contour)
        # print('\npoly\n', poly)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()

        # print('\nsegmentation\n', segmentation)
        segmentations.append(segmentation)
    # print('\nsegmentations\n', segmentations)
    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    # print('\nmin_x, min_y, max_x, max_y', x, y, max_x, max_y)
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area
    # print('\narea\n', area)
    contour_point_num = 0
    for sub_list in segmentations:
        contour_point_num = contour_point_num + len(sub_list)
    # print('contour_point_num', contour_point_num)
    '''

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation

print('creating directories if it does not exist')
#create destination directory if it does not exist
make_check_dir(dest_path)
#create Original and Ground_Truth the destination path
for dir_name in dest_subdir_name :
    make_check_dir(os.path.join(dest_path, dir_name))


(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(data_file_path)
images_list = sorted(glob(os.path.join(source_path, "Original\\*")))
masks_list = sorted(glob(os.path.join(source_path, "Ground_Truth\\*")))

# All the file names for images and mask should be identical in both name and order
assert [os.path.basename(x) for x in images_list] == [os.path.basename(x) for x in masks_list], 'All the file names for images and mask should be identical in both name and order'


print('copying Originals')
#copy Originals without any modifications using shutil
time.sleep(0.1) # without delay the progress bar becomes a bit messed up
for file_path in tqdm.tqdm(images_list):
    copy_path = os.path.join(dest_path, dest_subdir_name[0], os.path.basename(file_path))
    shutil.copy(file_path, copy_path)

print('Collect Original image data for COCO dataset')
coco_image_list = []
#mock image dictionary from COCO dataset
#{"id":0,"license":1,"file_name":"94_png.rf.af4c070c32db598db1bee46b3503c07b.jpg","height":640,"width":640,"date_captured":"2023-01-15T12:09:02+00:00"}
time.sleep(0.1)
image_total_iter = len(images_list)
license_id = 1 #license id for the images
with tqdm.tqdm(total=image_total_iter) as pbar:
    for i in range(image_total_iter):
        file_path = images_list[i]
        file_base_name = os.path.basename(file_path)
        original_image = cv2.imread(file_path)
        (original_image_height, original_image_width, _) = np.shape(original_image)
        # Contain time elapsed since EPOCH in float
        ti_c = os.path.getctime(file_path)
        ti_m = os.path.getmtime(file_path)
        # Converting the time in seconds to a timestamp
        c_ti = time.ctime(ti_c)
        m_ti = time.ctime(ti_m)
        image_dict = {
            'id': i,
            'license': license_id, #just set uniform license placeholder for now
            'file_name': file_base_name,
            'height': original_image_height,
            'width': original_image_width,
            'date_captured': c_ti # Use file creation date instead of last modification date
        }
        coco_image_list.append(image_dict)
        pbar.update(1)



print('Running preprocessing (clustering objects in image) and generating COCO dataset')
#copy masks and also do processing to produce object annotation files(numpy array, .npy) that stores object mask,
#copy Ground_Truth without any modifications using shutil
coco_annotation_list = []
total_iter = len(masks_list)
print(total_iter)
time.sleep(0.1) #delay added to stop print from interfering with tqdm progress bar
annotation_id = 0
is_crowd = 0
category_id = 1 # 0 is for background and 1 is for text, as all annotation is text fix id to 1
with tqdm.tqdm(total=total_iter) as pbar:
    for i in range(total_iter):
        file_path = masks_list[i]
        file_base_name = os.path.basename(file_path)
        original_mask = cv2.imread(file_path)
        is_mask_black = (cv2.countNonZero(original_mask[:, :, 0]) == 0)  # just check the first channel as cv2.countNonZero works only for single channel images
        if is_mask_black:  # mask/Ground_Truth is completely black
            if GEN_COLORED_MASK:  # if mask is black(i.e. empty) no need to generate colored mask just copy empty file
                shutil.copy(file_path, os.path.join(dest_path, dest_subdir_name[1], file_base_name))
            pbar.update(1)
        else:
            box_masks, boxes, w, h = extract_bboxes(file_path)
            if GEN_COLORED_MASK:
                colored_mask = colored_mask_generator(original_mask, box_masks, boxes)
                # dest_path//dest_subdir_name[1](Colored_Mask)//file_name.png
                cv2.imwrite(os.path.join(dest_path, dest_subdir_name[1], file_base_name), colored_mask)

            num_objects = len(boxes)


            for object_num in range(num_objects):
                object_annotation = create_annotation_from_partial_mask(box_masks[object_num], boxes[object_num], image_id=i, category_id=category_id, annotation_id=annotation_id, is_crowd=is_crowd)
                if object_annotation['area'] == 0 or np.isnan(object_annotation['bbox']).any():
                    # when object has area of 0, invalid representation of object skip from consideration
                    # or if the object has invalid bbox(containing NaN instead of ints)
                    pass
                else:
                    # remove empty list from 'segmentation' list of lists
                    object_annotation['segmentation'] = list(filter(lambda x: x != [], object_annotation['segmentation']))
                    annotation_id = annotation_id + 1
                    coco_annotation_list.append(object_annotation)
            pbar.update(1)
coco_info = \
    {
        "year": datetime.date.today().year, # 2023
        "version": 1,
        "description": 'A COCO dataset generated using COCO_json_segmentation_dataset_generation_from_binary_mask.py from original directory {}'.format(source_path),
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