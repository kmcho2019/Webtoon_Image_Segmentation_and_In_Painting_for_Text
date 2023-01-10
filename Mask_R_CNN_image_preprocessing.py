# Used to pre-proces image for Mask R-CNN Model
# Each black and white mask needs to be divided by objects and each objects needs a boudning box
# This program performs such operations and stores the operation in an npy file
# Uses xmeans from pyclustering to generate each object in a mask

import os
import shutil

import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random

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

data_file_path = r'.\combined_dataset_checked_additonal_exclusions_1'#r'.\combined_dataset_checked'
new_processed_data_path = data_file_path + '_Mask_R_CNN_preprocessed'
np.random.seed(42)
INIT_CENTERS = 1
MAX_CENTERS = 10
MIN_DISTANCE = 3 #any distance greater than MIN_DISTANCE will result in separate objects

sub_directory_list = ['train_dir', 'valid_dir', 'test_dir']
sub_sub_directory_list = ['Ground_Truth', 'Original']

# Delete Directory if is previously exists (in order to prevent previous runs messing with current run of program)
if os.path.exists(new_processed_data_path):
    shutil.rmtree(new_processed_data_path)
    print(f"{new_processed_data_path} has been deleted")
else:
    print(f"{new_processed_data_path} does not exist.")

def make_check_dir(check_file_path):
    isExist = os.path.exists(check_file_path)
    if isExist == False:
        os.makedirs(check_file_path)

def load_data(path, split=0.1):
    images = sorted(glob(os.path.join(path, "Original\\*")))
    masks = sorted(glob(os.path.join(path, "Ground_Truth\\*")))
    # images = sorted(glob(os.path.join(path, "images/*")))
    # masks = sorted(glob(os.path.join(path, "masks/*")))

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

print('creating directories')
#create test_dir, valid_dir, and test_dir in new path and also create Original and Ground_Truth in each of the three
for dir_name in sub_directory_list:
    make_check_dir(new_processed_data_path + '\\' + dir_name)
    for sub_dir in sub_sub_directory_list:
        make_check_dir(new_processed_data_path + '\\' + dir_name + '\\' + sub_dir)

(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(data_file_path)
original_files_list = [train_x, valid_x, test_x]
print('number of valid training images: ', len(train_x))
print('number of valid validation images: ', len(valid_x))
print('number of valid testing images: ', len(test_x))
print('Total number of valid images: ', len(train_x) + len(valid_x) + len(test_x))
print('number of valid training masks: ', len(train_y))
print('number of valid validation masks: ', len(valid_y))
print('number of valid testing masks: ', len(test_y))
print('Total number of valid masks: ', len(train_y) + len(valid_y) + len(test_y))
print('copying Originals')
#copy Originals without any modifications using shutil
for i, dir_name in enumerate(sub_directory_list):
    original_file_paths = original_files_list[i]
    copy_path = new_processed_data_path + '\\' + dir_name + '\\' + sub_sub_directory_list[1] #'Original'
    print('Copying files in ' + dir_name)
    time.sleep(0.1) # without delay the progress bar becomes a bit messed up
    for file_path in tqdm.tqdm(original_file_paths):
        shutil.copy(file_path, copy_path)

print('Copying Ground_Truth and Running Preprocessing')
#copy masks and also do processing to produce object annotation files(numpy array, .npy) that stores object mask,
#copy Ground_Truth without any modifications using shutil
for i, dir_name in enumerate(sub_directory_list):
    print('Current Directory: ' + str(dir_name))
    print('Directory Progress: ' + str(i+1) + '/' + str(len(sub_directory_list)))
    original_file_paths = original_files_list[i]
    copy_path = new_processed_data_path + '\\' + dir_name + '\\' + sub_sub_directory_list[0] #'Ground_Truth'
    with tqdm.tqdm(total=len(original_file_paths)) as pbar:
        for j, file_path in tqdm.tqdm(enumerate(original_file_paths), position=0, leave=True):
            # print('Current Directory: ' + str(dir_name))
            # print('Image Progress: ' + str(j+1) + '/' + str(len(original_file_paths)))
            shutil.copy(file_path, copy_path) #copy mask file

            #file_name = os.path.basename(file_path)
            # print('file_path', file_path)
            file_name_with_extension = os.path.basename(file_path)
            file_name, file_extension = os.path.splitext(file_name_with_extension)
            # print('file_name', file_name)
            #file_id = file_name[:-4] #remove the '.png' from file_name
            box_masks, boxes, w, h = extract_bboxes(file_path)
            num_objects = len(boxes)

            #data storage format for each object
            # [[min_x, min_y, max_x, max_y], (box_mask 2D numpy array)] => unit
            # box mask 2D array consists of shape  (row, col) => ((max_y - min_y + 1),(max_x - min_x + 1)) 0 for background 1 for text
            # each image has format of np.array([[image_w, image_h], unit1, unit2, unit3, ...]) => image_list
            # number of objects in one image == len(image_list) - 1 (first element of list contains image width and height)
            image_list = []
            image_list.append([w, h])
            for object_num in range(num_objects):
                unit_list = []
                [min_x, min_y, max_x, max_y] = boxes[object_num]
                bounding_box_list = np.array([min_x, min_y, max_x, max_y])
                box_mask = box_masks[object_num]
                unit_list.append(bounding_box_list)
                unit_list.append(box_mask)
                image_list.append(unit_list)
            numpy_array_image = np.array(image_list, dtype=object)
            numpy_array_image_name = copy_path + '\\' + file_name + '_annotation.npy'
            np.save(numpy_array_image_name, numpy_array_image)
            pbar.update(1)
