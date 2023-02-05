import os
import subprocess
import cv2
import numpy as np
import math
import glob
import datetime

run_magick_script = False
use_combine_panel_mode = False #try to process data even if clean original panel number is different by combining them, tends to produce pretty bad data
print('run_magick_script', run_magick_script)
resize_width = 512 #256
original_file_path = r'.\combined_original'#r'.\VLT_original_100_105'#r'C:\Users\kmcho\OneDrive - postech.ac.kr\바탕 화면\2022_2_Semester\Signals_and_Systems_EECE233\Research_Project\Test_images\VLT_original_100_105'
clean_file_path = r'.\combined_clean'#r'.\VLT_clean_100_105'#r'C:\Users\kmcho\OneDrive - postech.ac.kr\바탕 화면\2022_2_Semester\Signals_and_Systems_EECE233\Research_Project\Test_images\VLT_clean_100_105'
diff_file_path = r'.\combined_diff' #r'.\VLT_diff_100_105'#r'C:\Users\kmcho\OneDrive - postech.ac.kr\바탕 화면\2022_2_Semester\Signals_and_Systems_EECE233\Research_Project\Test_images\VLT_diff_100_105'
diff_slice_file_path = r'.\combined_diff_slice'#r'.\VLT_diff_slice_100_105'#r'C:\Users\kmcho\OneDrive - postech.ac.kr\바탕 화면\2022_2_Semester\Signals_and_Systems_EECE233\Research_Project\Test_images\VLT_diff_slice_100_105'
original_resize_file_path = r'.\combined_original_resize' #r'.\VLT_original_224_resize_100_105'
combined_file_path = r'.\combined_dataset' #r'.\VLT_combined_224_resize_100_105' #stores all of the files in image form, directories Original (for resized original panel) and Grount Truth (for difference)
directory_name_list = ['100', '101', '102', '103', '104', '105']
diff_slice_file_path = r'.\combined_diff_slice_size_512'#r'.\VLT_diff_slice_100_105'#r'C:\Users\kmcho\OneDrive - postech.ac.kr\바탕 화면\2022_2_Semester\Signals_and_Systems_EECE233\Research_Project\Test_images\VLT_diff_slice_100_105'
original_resize_file_path = r'.\combined_original_resize_size_512' #r'.\VLT_original_224_resize_100_105'
combined_file_path = r'.\combined_dataset_size_512' #r'.\VLT_combined_224_resize_100_105' #stores all of the files in image form, directories Original (for resized original panel) and Grount Truth (for difference)

output_image_name = directory_name_list[0]

original_dir_list = os.listdir(original_file_path)
clean_dir_list = os.listdir(clean_file_path)
if original_dir_list == clean_dir_list:
    directory_name_list = original_dir_list
    print(directory_name_list)
else:
    print('Directory of Original and Clean Does Not Match!')
    exit()

def make_check_dir(check_file_path):
    isExist = os.path.exists(check_file_path)
    if isExist == False:
        os.makedirs(check_file_path)

# processes image by passing it through filters and reshaping it, also reshapes the original
def image_processing_pipeline(f_image_name, f_original_image_name, f_resize_width=resize_width):
    f_img = cv2.imread(f_image_name)  # , cv2.IMREAD_UNCHANGED)
    f_original_img = cv2.imread(f_original_image_name)

    np.shape(f_img)
    print(np.shape(f_img))
    f_img_height = np.shape(f_img)[0]
    f_img_width = np.shape(f_img)[1]
    print('img_height', f_img_height)
    print('img_width', f_img_width)
    f_slice_num = math.ceil(f_img_height / f_img_width)  # number of vertical slices per image
    print('slice_num', f_slice_num)
    f_resize_height = round((f_img_height / f_img_width) * f_resize_width)
    print('resize_height', f_resize_height)

    f_resize_dim = (f_resize_width, f_resize_height)

    # #full new processing pipeline (uses cv2.INTER_AREA method and grayscaling)
    f_grayscale_img = f_img
    ceil_to_255 = np.vectorize(lambda t: 255 if t > 0 else 0)
    filter_20_to_255 = np.vectorize(lambda t: 255 if t > 20 else 0)
    filter_10_to_255 = np.vectorize(lambda t: 255 if t > 10 else 0)
    f_ceil_grayscale_img = filter_20_to_255(f_grayscale_img)
    f_blur_ceil_grayscale_img = cv2.blur(f_ceil_grayscale_img, (2, 2))
    f_blur_ceil_grayscale_img = filter_10_to_255(f_blur_ceil_grayscale_img)
    f_blur_ceil_grayscale_img = np.array(f_blur_ceil_grayscale_img, dtype='uint8')  # uint8 conversion needed for resize
    f_resize_blur_ceil_grayscale_img = cv2.resize(f_blur_ceil_grayscale_img, f_resize_dim, interpolation=cv2.INTER_AREA)
    f_resize_blur_ceil_grayscale_img = filter_10_to_255(f_resize_blur_ceil_grayscale_img)
    f_blur_resize_img = f_resize_blur_ceil_grayscale_img
    f_original_resize_img = cv2.resize(f_original_img, f_resize_dim,
                                     interpolation=cv2.INTER_AREA)  # cv2.INTER_CUBIC
    return f_blur_resize_img, f_original_resize_img, f_slice_num, f_resize_height


e = datetime.datetime.now()
current_datetime_str = e.strftime('%Y_%m_%d_%H_%M_%S') #used to create different temp files

make_check_dir(diff_file_path)
make_check_dir(diff_slice_file_path)
make_check_dir(original_resize_file_path)
make_check_dir(combined_file_path)
for dir_name in directory_name_list:
    make_check_dir(diff_file_path + '\\'+ dir_name)
    make_check_dir(diff_slice_file_path + '\\'+ dir_name)
    make_check_dir(original_resize_file_path + '\\'+ dir_name)
make_check_dir(combined_file_path + '\\' + 'Ground_Truth')
make_check_dir(combined_file_path + '\\' + 'Original')

total_resized_original_image_count = 0
total_mask_count = 0

for dir_num in range(len(directory_name_list)):
    dir_path = original_file_path + '\\' + directory_name_list[dir_num]
    original_file_count = 0
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            original_file_count += 1
    print('Original File count:', original_file_count)


    # Get a list of all the PNG files in the current directory
    clean_dir_path = clean_file_path + '\\' + directory_name_list[dir_num]
    filenames = glob.glob(clean_dir_path + '\\' + '*.png')

    # Sort the filenames so that they are in numerical order
    filenames.sort()

    clean_file_count = len(filenames)

    # if the file counts do not match try to stitch together all the panels in one chapter and then run script instead of going panel by panel
    if clean_file_count != original_file_count:
        if use_combine_panel_mode == False:
            print('Combine Panel Mode Disabled Change settings or remove directory({}).'.format(directory_name_list[dir_num]))
            exit()
        print('Combined panel case triggered')
        print('clean_file_count:', clean_file_count)
        print('original_file_count:', original_file_count)
        original_filenames = glob.glob(dir_path + '\\' + '*.png')
        original_filenames.sort()
        temp_directory_path = r'.\temp_' + current_datetime_str
        make_check_dir(temp_directory_path) #make temp directory to stores the stiched together panels
        temp_directory_path_current_dir = temp_directory_path + '\\' + directory_name_list[dir_num]
        make_check_dir(temp_directory_path_current_dir)



        # Load the images and store them in a list
        images = [cv2.imread(filename) for filename in filenames]
        original_images = [cv2.imread(filename) for filename in original_filenames]

        # Calculate the size of the final image
        image_height = sum([image.shape[0] for image in images])
        image_width = max([image.shape[1] for image in images])
        image_shape = (image_height, image_width, 3)

        original_image_height = sum([image.shape[0] for image in original_images])
        original_image_width = max([image.shape[1] for image in original_images])
        original_image_shape = (original_image_height, original_image_width, 3)
        print('Full vertical clean image shape', image_shape)
        print('Full vertical original image shape', original_image_shape)
        #check if the stitched together images match between clean and original
        if abs(image_height - original_image_height) > 10 or abs(image_width - original_image_width) > 5:
            print('ERROR!, Mismatch between full clean and original images, check data!')
            exit()
        # Create an empty image to hold the final result
        result = np.zeros(image_shape, dtype=np.uint8)

        original_result = np.zeros(original_image_shape, dtype=np.uint8)

        # Starting position for the next image
        y = 0
        original_y = 0

        # Iterate through the images and stitch them together
        for image in images:
            result[y:y + image.shape[0], 0:image.shape[1]] = image
            y += image.shape[0]

        for image in original_images:
            original_result[original_y:original_y + image.shape[0], 0:image.shape[1]] = image
            original_y += image.shape[0]

        # Save the result
        temp_original_file_name = temp_directory_path_current_dir + '\\' + 'original.png'
        temp_clean_file_name = temp_directory_path_current_dir + '\\' + 'clean.png'
        cv2.imwrite(temp_clean_file_name, result)
        cv2.imwrite(temp_original_file_name, original_result)
        diff_file_name = diff_file_path + '\\' + directory_name_list[dir_num] + '\\' + '001.png' #only one exists as all of the image is combined into one
        if run_magick_script:
            script ='magick ' + temp_original_file_name + ' ' + temp_clean_file_name + ' -compose difference -composite -evaluate Pow 2 -evaluate divide 3 -separate -evaluate-sequence Add -evaluate Pow 0.5 ' + diff_file_name
            print('script:\n')
            print(script)
            l = script.split()
            # os.system(script)
            return_val = subprocess.run(l)
            print('return_val', return_val)
        #image processing pipeline (take combined panel apply filters, reshape them, take slices of them to produce


        original_image_name = temp_original_file_name #orignal that is used to generate resized version of them (in this case one very long image)
        image_name = diff_file_name #diff file that comes from image magick script output

        blur_resize_img, original_resize_img,slice_num,resize_height = image_processing_pipeline(image_name, original_image_name, resize_width)

        np.shape(blur_resize_img)

        for j in range(slice_num):
            if j == (slice_num - 1): #skip last slice combined panel as it contains logos which are not cleaned
                break
            else:
                crop_img = blur_resize_img[resize_width * j: resize_width * (j + 1), :]
                original_crop_img = original_resize_img[resize_width * j: resize_width * (j + 1), :]

            print(j, 'crop_img size', np.shape(crop_img))

            if j == 0:
                l = np.expand_dims(crop_img, axis=0)
                l_original = np.expand_dims(original_crop_img, axis=0)
            else:
                l = np.append(l, np.expand_dims(crop_img, axis=0), axis=0)
                l_original = np.append(l_original, np.expand_dims(original_crop_img, axis=0), axis=0)
            # l.append(crop_img)
            print('list size', np.shape(l))
            print('original resize list size', np.shape(l_original))
        np.shape(l)
        np.shape(l_original)
        #just give one npy file as it uses one combined panel
        npy_file_name = diff_slice_file_path + '\\' + directory_name_list[dir_num] + '\\' + '001.npy'#r'{:03n}.npy'.format(i+1)
        original_npy = original_resize_file_path+ '\\' + directory_name_list[dir_num] + '\\' + '001.npy'#r'{:03n}.npy'.format(i+1)
        with open(npy_file_name, 'wb') as f:
            np.save(f, l)
        with open(original_npy, 'wb') as f:
            np.save(f, l_original)
        for j in range(slice_num):
            if j == (slice_num - 1):
                break

            total_resized_original_image_count = total_resized_original_image_count + 1
            total_mask_count = total_mask_count + 1
            combined_dir_mask_name = combined_file_path + r'\Ground_Truth' + '\\' + str(total_mask_count) +'.png'
            combined_dir_original_crop_name = combined_file_path + r'\Original' + '\\' + str(total_resized_original_image_count) +'.png'
            file_name = diff_slice_file_path + '\\' + directory_name_list[dir_num] + \
                        '\\' + r'{:03n}_'.format(0+1) + r'diff_slice_{:03n}.png'.format(j+1) # i replaced by 0 in .format(0+1)
            original_crop_file_name = original_resize_file_path + '\\' + directory_name_list[dir_num] + \
                                      '\\' + r'{:03n}_'.format(0+1) + r'resize_slice_{:03n}.png'.format(j+1) # i replaced by 0 in .format(0+1)

            print(j, file_name)
            print(j, original_crop_file_name)
            # print(j, combined_dir_mask_name)
            # print(j, combined_dir_original_crop_name)
            cv2.imwrite(file_name, l[j])
            cv2.imwrite(original_crop_file_name, l_original[j])
            cv2.imwrite(combined_dir_mask_name, l[j])
            cv2.imwrite(combined_dir_original_crop_name, l_original[j])

        # no need for if function as i == 0 with combined panel
        dir_original_reshape_collection = l_original
        dir_diff_collection = l

        print('dir_original_reshape_collection', np.shape(dir_original_reshape_collection))
        print('dir_diff_collection', np.shape(dir_diff_collection))

    else:
        original_file_list = glob.glob(original_file_path + '\\' + directory_name_list[dir_num] + '\\' + '*.png')
        clean_file_list = glob.glob(clean_file_path + '\\' + directory_name_list[dir_num] + '\\' + '*.png')
        original_file_list.sort() #ascending sort
        clean_file_list.sort() #ascending sort
        # print(directory_name_list[dir_num])
        assert len(original_file_list) == len(clean_file_list), 'original and clean png number does not match at directory({})'.format(directory_name_list[dir_num])
        #when original file count matches that of clean file count
        #execute for each file
        for i in range(original_file_count):
            #create diff file by comparing original and clean
            print('original_file_list', original_file_list)
            print('clean_file_list', clean_file_list)
            print('i', i)
            diff_file_name = r'{:03n}.png'.format(i+1)
            original_file_name = original_file_list[i]
            clean_file_name = clean_file_list[i]
            original_sub_dir_path = '\\' + directory_name_list[dir_num] + '\\' + original_file_name
            clean_sub_dir_path = '\\' + directory_name_list[dir_num] + '\\' + clean_file_name
            diff_sub_dir_path = '\\' + directory_name_list[dir_num] + '\\' + diff_file_name
            # print(file_path)
            if run_magick_script:
                older_script = 'magick compare ' + original_file_name + ' ' + clean_file_name + ' -compose Src -highlight-color White -lowlight-color Black ' + diff_file_path + diff_sub_dir_path #old processing command
                #new script uses gray scale comparison so that a more nuanced comparison and isolation of the region is possible
                old_script = 'magick ' + original_file_name + ' ' + clean_file_name + ' -compose difference -composite -colorspace Gray ' + diff_file_path + diff_sub_dir_path #less accurate color comparison(grayscale), also does not work well
                script ='magick ' + original_file_name + ' ' + clean_file_name + ' -compose difference -composite -evaluate Pow 2 -evaluate divide 3 -separate -evaluate-sequence Add -evaluate Pow 0.5 ' + diff_file_path + diff_sub_dir_path
                print('script:\n')
                print(script)
                l = script.split()
                # os.system(script)
                return_val = subprocess.run(l)
                print('return_val', return_val)

            #create diff_slice file by processing slice
            #taken from Image_processing.ipynb

            original_image_name = original_file_name
            image_name = diff_file_path + diff_sub_dir_path

            #new processing pipeline
            blur_resize_img, original_resize_img, slice_num,resize_height = image_processing_pipeline(image_name, original_image_name, resize_width)

            #old processing pipeline

            # original_blur_img = cv2.blur(img, (2,2))  # blur needs to be applied as the difference is somewhat noisy so it is required to avoid pixelating the
            # # binary_original_blur_img = np.ceil(original_blur_img)
            # ceil_to_255 = np.vectorize(lambda t: 255 if t > 0 else 0)
            # #binary_original_blur_img = ceil_to_255(original_blur_img)
            # blur_resize_img = cv2.resize(original_blur_img, resize_dim,
            #                              interpolation=cv2.INTER_LANCZOS4)  # cv2.INTER_CUBIC
            # blur_resize_img = ceil_to_255(blur_resize_img)
            #
            # original_resize_img = cv2.resize(original_img, resize_dim,
            #                              interpolation=cv2.INTER_LANCZOS4)  # cv2.INTER_CUBIC
            np.shape(blur_resize_img)
            print('i, original_file_count, slice_num',i, original_file_count, slice_num)
            for j in range(slice_num):
                print('i, original_file_count, j, slice_num',i, original_file_count, j, slice_num)
                if i != (original_file_count-1) and j == (slice_num - 1):
                    crop_img = blur_resize_img[(resize_height - resize_width):, :]
                    original_crop_img = original_resize_img[(resize_height - resize_width):, :]
                elif i == (original_file_count - 1) and j == (slice_num - 1): #skip last slice of last panel of chapter as it contains logos which are not cleaned
                    print('SKIP!')
                    break
                else:
                    crop_img = blur_resize_img[resize_width * j: resize_width * (j + 1), :]
                    original_crop_img = original_resize_img[resize_width * j: resize_width * (j + 1), :]

                print(j, 'crop_img size', np.shape(crop_img))

                if j == 0:
                    l = np.expand_dims(crop_img, axis=0)
                    l_original = np.expand_dims(original_crop_img, axis=0)
                else:
                    l = np.append(l, np.expand_dims(crop_img, axis=0), axis=0)
                    l_original = np.append(l_original, np.expand_dims(original_crop_img, axis=0), axis=0)
                # l.append(crop_img)
                print('list size', np.shape(l))
                print('original resize list size', np.shape(l_original))
            np.shape(l)
            np.shape(l_original)
            npy_file_name = diff_slice_file_path + '\\' + directory_name_list[dir_num] + '\\' + r'{:03n}.npy'.format(i+1)
            original_npy = original_resize_file_path+ '\\' + directory_name_list[dir_num] + '\\' + r'{:03n}.npy'.format(i+1)
            with open(npy_file_name, 'wb') as f:
                np.save(f, l)
            with open(original_npy, 'wb') as f:
                np.save(f, l_original)
            for j in range(slice_num):
                print('i, original_file_count, j, slice_num',i, original_file_count, j, slice_num)
                if i == (original_file_count -1) and j == (slice_num - 1):
                    print('SKIP!')
                    break
                total_resized_original_image_count = total_resized_original_image_count + 1
                total_mask_count = total_mask_count + 1
                combined_dir_mask_name = combined_file_path + r'\Ground_Truth' + '\\' + str(total_mask_count) +'.png'
                combined_dir_original_crop_name = combined_file_path + r'\Original' + '\\' + str(total_resized_original_image_count) +'.png'
                file_name = diff_slice_file_path + '\\' + directory_name_list[dir_num] + \
                            '\\' + r'{:03n}_'.format(i+1) + r'diff_slice_{:03n}.png'.format(j+1)
                original_crop_file_name = original_resize_file_path + '\\' + directory_name_list[dir_num] + \
                                          '\\' + r'{:03n}_'.format(i+1) + r'resize_slice_{:03n}.png'.format(j+1)

                print(j, file_name)
                print(j, original_crop_file_name)
                # print(j, combined_dir_mask_name)
                # print(j, combined_dir_original_crop_name)
                cv2.imwrite(file_name, l[j])
                cv2.imwrite(original_crop_file_name, l_original[j])
                cv2.imwrite(combined_dir_mask_name, l[j])
                cv2.imwrite(combined_dir_original_crop_name, l_original[j])
            if i == 0:
                dir_original_reshape_collection = l_original
                dir_diff_collection = l
            else:
                print('DEBUG dir_original_reshape_collection', np.shape(dir_original_reshape_collection))
                print('DEBUG dir_diff_collection', np.shape(dir_diff_collection))
                print('DEBUG l_original',np.shape(l_original))
                print('DEBUG l', np.shape(np.array(l)))
                print('DEBUG i',i)
                #append only when size is correct
                l_original = np.array(l_original)
                l = np.array(l)
                if l_original.ndim >=2 and l_original.shape[1:] == (resize_width, resize_width, 3) and l.ndim >=2 and l.shape[1:] == (resize_width, resize_width, 3):
                    dir_original_reshape_collection = np.append(dir_original_reshape_collection, l_original, axis=0)
                    dir_diff_collection = np.append(dir_diff_collection, l, axis=0)
            print('dir_original_reshape_collection', np.shape(dir_original_reshape_collection))
            print('dir_diff_collection', np.shape(dir_diff_collection))


    print(directory_name_list[dir_num])
    print('reshape_collection', np.shape(dir_original_reshape_collection))
    print('diff_collection', np.shape(dir_diff_collection))

    dir_original_npy_name = original_resize_file_path + '\\' + directory_name_list[dir_num] + '\\' + 'reshape_collection_' + directory_name_list[dir_num] + '.npy'
    dir_diff_npy_name = diff_slice_file_path + '\\' + directory_name_list[dir_num] + '\\' + 'diff_collection_' + directory_name_list[dir_num] + '.npy'
    with open(dir_original_npy_name, 'wb') as f:
        np.save(f, dir_original_reshape_collection)
    with open(dir_diff_npy_name, 'wb') as f:
        np.save(f, dir_diff_collection)

    if dir_num == 0:
        test_original_reshape_collection = dir_original_reshape_collection
        test_diff_collection = dir_diff_collection
    elif dir_num == 1:
        total_original_reshape_collection = dir_original_reshape_collection
        total_diff_collection = dir_diff_collection
        print('total_original_reshape_collection', np.shape(total_original_reshape_collection))
        print('total_diff_collection', np.shape(total_diff_collection))
    else:
        total_original_reshape_collection = np.append(total_original_reshape_collection, dir_original_reshape_collection, axis=0)
        total_diff_collection = np.append(total_diff_collection, dir_diff_collection, axis=0)
        print('total_original_reshape_collection', np.shape(total_original_reshape_collection))
        print('total_diff_collection', np.shape(total_diff_collection))


print('test_original_reshape_collection', np.shape(test_original_reshape_collection))
print('test_diff_collection', np.shape(test_diff_collection))

test_original_npy_name = original_resize_file_path + '\\' + 'test_reshape_collection.npy'
test_diff_npy_name = diff_slice_file_path + '\\' + 'test_diff_collection.npy'
total_original_npy_name = original_resize_file_path + '\\' + 'total_reshape_collection.npy'
total_diff_npy_name = diff_slice_file_path + '\\' + 'total_diff_collection.npy'
if len(directory_name_list) > 1: #as the very first entry goes to test_reshape/diff_collection so need 2 or more directories
    with open(total_original_npy_name, 'wb') as f:
        np.save(f, total_original_reshape_collection)
    with open(total_diff_npy_name, 'wb') as f:
        np.save(f, total_diff_collection)

with open(test_original_npy_name, 'wb') as f:
    np.save(f, test_original_reshape_collection)
with open(test_diff_npy_name, 'wb') as f:
    np.save(f, test_diff_collection)
'''        
file_path = r'\100\001.png'
script = 'magick compare ' + original_file_path + file_path + ' '+ clean_file_path + file_path + ' -compose Src -highlight-color White -lowlight-color Black ' + diff_file_path + file_path
print('script:\n')
print(script)
l = script.split()
# os.system(script)
return_val = subprocess.run(l)
print('return_val', return_val)
'''

#
# script = 'magick compare ' + file_path + 'clean.png' + file_path + 'original.png' + '-compose Src -highlight-color White -lowlight-color Black' + output_image_name
# l = script.split()
# # os.system(script)
# return_val = subprocess.run(l)
# print('return_val', return_val)
