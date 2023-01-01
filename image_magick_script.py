import os
import subprocess
import cv2
import numpy as np
import math

run_magick_script = True
print('run_magick_script', run_magick_script)
resize_width = 256
original_file_path = r'.\combined_original'#r'.\VLT_original_100_105'#r'C:\Users\kmcho\OneDrive - postech.ac.kr\바탕 화면\2022_2_Semester\Signals_and_Systems_EECE233\Research_Project\Test_images\VLT_original_100_105'
clean_file_path = r'.\combined_clean'#r'.\VLT_clean_100_105'#r'C:\Users\kmcho\OneDrive - postech.ac.kr\바탕 화면\2022_2_Semester\Signals_and_Systems_EECE233\Research_Project\Test_images\VLT_clean_100_105'
diff_file_path = r'.\combined_diff' #r'.\VLT_diff_100_105'#r'C:\Users\kmcho\OneDrive - postech.ac.kr\바탕 화면\2022_2_Semester\Signals_and_Systems_EECE233\Research_Project\Test_images\VLT_diff_100_105'
diff_slice_file_path = r'.\combined_diff_slice'#r'.\VLT_diff_slice_100_105'#r'C:\Users\kmcho\OneDrive - postech.ac.kr\바탕 화면\2022_2_Semester\Signals_and_Systems_EECE233\Research_Project\Test_images\VLT_diff_slice_100_105'
original_resize_file_path = r'.\combined_original_resize' #r'.\VLT_original_224_resize_100_105'
combined_file_path = r'.\combined_dataset' #r'.\VLT_combined_224_resize_100_105' #stores all of the files in image form, directories Original (for resized original panel) and Grount Truth (for difference)
directory_name_list = ['100', '101', '102', '103', '104', '105']
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
    count = 0
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    print('File count:', count)
    #execute for each file
    for i in range(count):
        #create diff file by comparing original and clean
        file_name = r'{:03n}.png'.format(i+1)
        print(file_name)
        file_path = '\\' + directory_name_list[dir_num] + '\\' + file_name
        # print(file_path)
        if run_magick_script:
            older_script = 'magick compare ' + original_file_path + file_path + ' ' + clean_file_path + file_path + ' -compose Src -highlight-color White -lowlight-color Black ' + diff_file_path + file_path #old processing command
            #new script uses gray scale comparison so that a more nuanced comparison and isolation of the region is possible
            old_script = 'magick ' + original_file_path + file_path + ' ' + clean_file_path + file_path + ' -compose difference -composite -colorspace Gray ' + diff_file_path + file_path #less accurate color comparison(grayscale), also does not work well
            script ='magick ' + original_file_path + file_path + ' ' + clean_file_path + file_path + ' -compose difference -composite -evaluate Pow 2 -evaluate divide 3 -separate -evaluate-sequence Add -evaluate Pow 0.5 ' + diff_file_path + file_path
            print('script:\n')
            print(script)
            l = script.split()
            # os.system(script)
            return_val = subprocess.run(l)
            print('return_val', return_val)

        #create diff_slice file by processing slice
        #taken from Image_processing.ipynb

        original_image_name = original_file_path + file_path
        image_name = diff_file_path + file_path

        img = cv2.imread(image_name)  # , cv2.IMREAD_UNCHANGED)
        original_img = cv2.imread(original_image_name)

        np.shape(img)
        print(np.shape(img))
        img_height = np.shape(img)[0]
        img_width = np.shape(img)[1]
        print('img_height', img_height)
        print('img_width', img_width)
        slice_num = math.ceil(img_height / img_width)  # number of vertical slices per image
        print('slice_num', slice_num)
        resize_height = round((img_height / img_width) * resize_width)
        print('resize_height', resize_height)

        resize_dim = (resize_width, resize_height)

        # #full new processing pipeline (uses cv2.INTER_AREA method and grayscaling)
        grayscale_img = img
        ceil_to_255 = np.vectorize(lambda t: 255 if t > 0 else 0)
        filter_20_to_255 = np.vectorize(lambda t: 255 if t > 20 else 0)
        filter_10_to_255 = np.vectorize(lambda t: 255 if t > 10 else 0)
        ceil_grayscale_img = filter_20_to_255(grayscale_img)
        blur_ceil_grayscale_img = cv2.blur(ceil_grayscale_img, (2,2))
        blur_ceil_grayscale_img = filter_10_to_255(blur_ceil_grayscale_img)
        blur_ceil_grayscale_img = np.array(blur_ceil_grayscale_img, dtype='uint8') #uint8 conversion needed for resize
        resize_blur_ceil_grayscale_img = cv2.resize(blur_ceil_grayscale_img, resize_dim, interpolation = cv2.INTER_AREA)
        resize_blur_ceil_grayscale_img = filter_10_to_255(resize_blur_ceil_grayscale_img)
        blur_resize_img = resize_blur_ceil_grayscale_img

        original_resize_img = cv2.resize(original_img, resize_dim,
                                     interpolation=cv2.INTER_AREA)  # cv2.INTER_CUBIC
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

        for j in range(slice_num):
            if i != (count-1) and j == (slice_num - 1):
                crop_img = blur_resize_img[(resize_height - resize_width):, :]
                original_crop_img = original_resize_img[(resize_height - resize_width):, :]
            elif i == (count - 1) and j == (slice_num - 1): #skip last slice of last panel of chapter as it contains logos which are not cleaned
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
            if i == (count -1) and j == (slice_num - 1):
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
