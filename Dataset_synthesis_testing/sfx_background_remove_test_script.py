import numpy as np
import cv2
# import matplotlib.pyplot as plt
import os
import glob

def make_transparent_bw(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    mask = np.where(image == 0, 0, 255).astype(np.uint8)
    result = cv2.bitwise_and(image, mask)
    result = np.dstack((result, mask))
    return result

def remove_white(img_path):
    # Load the image using cv2
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    print(np.shape(img))
    # Convert the image to RGBA format
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

    # Create a mask where white pixels are set to 0
    mask = np.where(rgba == (255, 255, 255, 255), 0, 255).astype(np.uint8)

    # Multiply the RGBA channels by the mask
    rgba = np.dstack((rgba, mask))
    rgba = np.expand_dims(rgba, axis=2)
    result = rgba * mask[:, :, np.newaxis, :]

    print(np.shape(rgba))
    print(np.shape(mask[:, :, np.newaxis]))
    rgba = rgba * mask[:, :, np.newaxis]
   
    return rgba

def attempt(img_path):
    src = cv2.imread(img_path, cv2.IMREAD_COLOR)
    alpha = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print(src)
    print(np.shape(src))

    reverse_converter = np.vectorize(lambda x: 0 if x < 255 else 255)
    alpha = reverse_converter(alpha)
    alpha.astype(np.uint8)
    
    b, g, r = cv2.split(src)
    rgba = [b, g, r, alpha]

    dst = cv2.merge(rgba, 4)
    return dst

def remove_background_func(file_path):
    image_bgr = cv2.imread(file_path)
    ret, image_bgr = cv2.threshold(image_bgr, 170, 255, cv2.THRESH_BINARY)

    gray_scale = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    gray_scale = gray_scale.astype(np.uint8)
    # k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    # k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    # gray_scale = cv2.morphologyEx(gray_scale, cv2.MORPH_CLOSE, k)
    ret, gray_scale = cv2.threshold(gray_scale, 170, 255, cv2.THRESH_BINARY)
    gray_scale = np.repeat(gray_scale[:, :, np.newaxis], 3, axis=2)
    # print(np.shape(gray_scale))
    # gray_scale = cv2.merge([gray_scale, gray_scale, gray_scale])
    image_bgr[gray_scale == 255] = 255
    # get the image dimensions (height, width and channels)
    h, w, c = image_bgr.shape
    # append Alpha channel -- required for BGRA (Blue, Green, Red, Alpha)
    image_bgra = np.concatenate([image_bgr, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
    # create a mask where white pixels ([255, 255, 255]) are True
    white = np.all(image_bgr == [255, 255, 255], axis=-1)
    # change the values of Alpha to 0 for all the white pixels
    image_bgra[white, -1] = 0
    # save the image
    # cv2. imwrite('test_dung_alt_bgra.png', image_bgra) #cv2.imwrite('image_bgra_2.png', image_bgra) # cv2.imwrite('image_bgra.png', image_bgra)
    return image_bgra

if __name__ == '__main__':
    # file_path = '.\\test_kwang.png'
    # file_path = '.\\test_dung.png'
    # file_path = '.\\brightness_check_3.png'
    # file_path = '.\\test_dung_alt.png'
    source_path = r'.\text_collection\unprocessed_foreground_collection'
    destination_path = r'.\text_collection\processed_foreground_collection'

    png_list = glob.glob(os.path.join(source_path, '*.png'))
    print(len(png_list))
    for png_file_path in png_list:
        file_name = os.path.basename(png_file_path)
        new_file_name = 'transparent_' + file_name
        new_file = remove_background_func(png_file_path)
        cv2.imwrite(os.path.join(destination_path, new_file_name), new_file)


'''
result = attempt(file_path)
print(np.shape(result))
plt.imshow(result)
plt.show()
cv2.imwrite(r'.\test_result.png', result)
'''