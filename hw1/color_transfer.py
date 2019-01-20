import cv2
import numpy as np
import sys

def convert_color_space_BGR_to_RGB(img_BGR):
    img_RGB = np.zeros_like(img_BGR,dtype=np.float32)
    #print(img_RGB.shape) (599,800,3)
    # to be completed ...
    #print("RGB:",img_RGB)
    #print("BGR:",img_BGR)
    img_RGB = img_BGR[:,:,::-1]
    #print("RGB:",img_RGB)
    #print(img_RGB.shape) i(599,800,3)
    return img_RGB

def convert_color_space_RGB_to_BGR(img_RGB):
    img_BGR = np.zeros_like(img_RGB,dtype=np.float32)
    # to be completed ...
    img_BGR = img_RGB[:,:,::-1]
    return img_BGR

def convert_color_space_RGB_to_Lab(img_RGB):
    '''
    convert image color space RGB to Lab
    '''
    img_LMS = np.zeros_like(img_RGB,dtype=np.float32)
    # to be completed ...

    img_Lab = np.zeros_like(img_RGB,dtype=np.float32)
    # to be completed ...
    return img_Lab

def convert_color_space_Lab_to_RGB(img_Lab):
    '''
    convert image color space Lab to RGB
    '''
    img_LMS = np.zeros_like(img_Lab,dtype=np.float32)
    # to be completed ...
    img_RGB = np.zeros_like(img_Lab,dtype=np.float32)
    # to be completed ...

    return img_RGB

def convert_color_space_RGB_to_CIECAM97s(img_RGB):
    '''
    convert image color space RGB to CIECAM97s
    '''
    img_CIECAM97s = np.zeros_like(img_RGB,dtype=np.float32)
    # to be completed ...

    return img_CIECAM97s

def convert_color_space_CIECAM97s_to_RGB(img_CIECAM97s):
    '''
    convert image color space CIECAM97s to RGB
    '''
    img_RGB = np.zeros_like(img_CIECAM97s,dtype=np.float32)
    # to be completed ...

    return img_RGB


def color_transfer_in_Lab(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_Lab =====')
    # to be completed ...
    new_rgb_img = convert_color_space_BGR_to_RGB(img_RGB_source)
    print(new_rgb_img)
    

def color_transfer_in_RGB(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_RGB =====')
    # to be completed ...

def color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_CIECAM97s =====')
    # to be completed ...

def color_transfer(img_RGB_source, img_RGB_target, option):
    if option == 'in_RGB':
        img_RGB_new = color_transfer_in_RGB(img_RGB_source, img_RGB_target)
    elif option == 'in_Lab':
        img_RGB_new = color_transfer_in_Lab(img_RGB_source, img_RGB_target)
    elif option == 'in_CIECAM97s':
        img_RGB_new = color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target)
    return img_RGB_new

if __name__ == "__main__":
    print('==================================================')
    print('PSU CS 410/510, Winter 2019, HW1: color transfer')
    print('==================================================')

    path_file_image_source = sys.argv[1]
    path_file_image_target = sys.argv[2]
 #   path_file_image_result_in_Lab = sys.argv[3]
 #   path_file_image_result_in_RGB = sys.argv[4]
 #   path_file_image_result_in_CIECAM97s = sys.argv[5]

    # ===== read input images
    img_RGB_source = cv2.imread(path_file_image_source) #is the image you want to change the its color
    img_RGB_target = cv2.imread(path_file_image_target) #is the image containing the color distribution that you want to change the img_RGB_source to (transfer color of the img_RGB_target to the img_RGB_source)
    img_RGB_new_Lab       = color_transfer(img_RGB_source, img_RGB_target, option='in_Lab')
    # todo: save image to path_file_image_result_in_Lab

    #img_RGB_new_RGB       = color_transfer(img_RGB_source, img_RGB_target, option='in_RGB')
    # todo: save image to path_file_image_result_in_RGB

    #img_RGB_new_CIECAM97s = color_transfer(img_RGB_source, img_RGB_target, option='in_CIECAM97s')
    # todo: save image to path_file_image_result_in_CIECAM97s

