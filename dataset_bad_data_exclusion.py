import os
import glob

# Uses a reference directory to conduct exclusions
USE_REFERENCE = False
# Set the source and destination directories

base_dir = r'.\combined_dataset_size_512'
ground_truth_dir = os.path.join(base_dir, r"Ground_Truth_Copy")
original_dir = os.path.join(base_dir, r"Original_Copy")
excluded_original_dir = os.path.join(base_dir, r"excluded_Original")
excluded_ground_truth_dir = os.path.join(base_dir, r"excluded_Ground_Truth")


# Reference already had its bad
reference_ground_truth_dir = r'.\combined_dataset_checked_additonal_exclusions_1\Ground_Truth'
reference_list = glob.glob(os.path.join(reference_ground_truth_dir, '*.png'))
filename_reference_set = set([os.path.basename(x) for x in reference_list])
print(filename_reference_set)
# Create the destination directory if it doesn't exist
if not os.path.exists(excluded_original_dir):
    os.makedirs(excluded_original_dir)

if not os.path.exists(excluded_ground_truth_dir):
    os.makedirs(excluded_ground_truth_dir)



# Get a set of the filenames in the Ground_Truth directory
ground_truth_filenames = set(os.listdir(ground_truth_dir))
original_filenames = set(os.listdir(original_dir))

if USE_REFERENCE:
    # Iterate through the files in the Ground_Truth directory
    for filename in os.listdir(ground_truth_dir):
        # If the file does not exist in reference move it to excluded directory
        if filename not in filename_reference_set:
            # Move the file to the excluded_Ground_Truth directory
            os.rename(os.path.join(ground_truth_dir, filename), os.path.join(excluded_ground_truth_dir, filename))

ground_truth_filenames = set(os.listdir(ground_truth_dir))
original_filenames = set(os.listdir(original_dir))

# Iterate through the files in the Original directory
for filename in os.listdir(original_dir):
    # Check if the file is a PNG file
    if not filename.endswith(".png"):
        continue

    # Check if the file exists in the Ground_Truth directory
    if filename not in ground_truth_filenames:
        # Move the file to the excluded_Original directory
        os.rename(os.path.join(original_dir, filename), os.path.join(excluded_original_dir, filename))

ground_truth_filenames = set(os.listdir(ground_truth_dir))
original_filenames = set(os.listdir(original_dir))

# Iterate through the files in the Ground_Truth directory
for filename in os.listdir(ground_truth_dir):
    # Check if the file is a PNG file
    if not filename.endswith(".png"):
        continue

    # Check if the file exists in the Original directory
    if filename not in original_filenames:
        # Move the file to the excluded_Original directory
        os.rename(os.path.join(ground_truth_dir, filename), os.path.join(excluded_ground_truth_dir, filename))