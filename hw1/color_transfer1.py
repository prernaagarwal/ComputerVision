# Prerna Agarwal
# Computer Vision CS410
# HW1, version 2
# Based on Color Transfer between Images research paper
# Kindly refer to the equations from the research paper
# mentioned above.

# Note: This program has incorrect matrix multiplication for RGB to LAB

import cv2
import numpy as np
import sys
import math

def convert_color_space_BGR_to_RGB(img_BGR):
    img_RGB = np.zeros_like(img_BGR,dtype=np.float32)
    B = img_BGR[:,:,0]
    G = img_BGR[:,:,1]
    R = img_BGR[:,:,2]
   
    img_RGB[:,:,0] = R
    img_RGB[:,:,1] = G
    img_RGB[:,:,2] = B
    #img_RGB = img_BGR[:,:,::-1]
    return img_RGB

def convert_color_space_RGB_to_BGR(img_RGB):
    img_BGR = np.zeros_like(img_RGB,dtype=np.float32)
    R = img_RGB[:,:,0]
    G = img_RGB[:,:,1]
    B = img_RGB[:,:,2]
   
    img_BGR[:,:,0] = B
    img_BGR[:,:,1] = G
    img_BGR[:,:,2] = R
    #img_BGR = img_RGB[:,:,::-1]
    return img_BGR

def convert_color_space_RGB_to_Lab(img_RGB):
    '''
    convert image color space RGB to Lab
    '''
    img_LMS = np.zeros_like(img_RGB,dtype=np.float32)
   
    LMStoRGB = np.array([[0.3811,0.5783,0.0402],[0.1967,0.7244,0.0782],[0.0241,0.1288,0.8444]])
    img_LMS = np.matmul(img_RGB,LMStoRGB)
    #print(img_LMS)

    img_LMS = np.log10(img_LMS)

    img_Lab = np.zeros_like(img_RGB,dtype=np.float32)
    matrix1 = [[1/math.sqrt(3),0,0],[0,1/math.sqrt(6),0],[0,0,1/math.sqrt(2)]]
    matrix2 = [[1,1,1],[1,1,-2],[1,-1,0]]
    matrix3 = np.matmul(matrix1,matrix2)
    img_Lab = np.matmul(img_LMS,matrix3)
     
    return img_Lab

def convert_color_space_Lab_to_RGB(img_Lab):
    '''
    convert image color space Lab to RGB
    '''
    img_LMS = np.zeros_like(img_Lab,dtype=np.float32)

    matrix1 = [[1,1,1],[1,1,-1],[1,-2,0]]
    matrix2 = [[math.sqrt(3)/3.0,0,0],[0,math.sqrt(6)/6.0,0],[0,0,math.sqrt(2)/2.0]]
    matrix3 = np.matmul(matrix1,matrix2)
    img_LMS = np.matmul(img_Lab, matrix3) 
    #img_LMS = np.matmul(matrix3, img_Lab)
    
    #take power here of Lms so 10 to the power LMS
    img_LMS = np.power(10,img_LMS)
    

    img_RGB = np.zeros_like(img_Lab,dtype=np.float32)

    matrix4 = [[4.4679, -3.5873, 0.1193],[-1.2186,2.3809,-0.1624],[0.0497, -0.2439, 1.2045]]
    img_RGB = np.matmul(img_LMS, matrix4)


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
    target_rgb = convert_color_space_BGR_to_RGB(img_RGB_target)
    
    new_src_img_Lab = convert_color_space_RGB_to_Lab(new_rgb_img)
    new_tar_img_Lab = convert_color_space_RGB_to_Lab(target_rgb) 


    combined_LAB = np.zeros_like(new_src_img_Lab,dtype=np.float32)

    l = new_src_img_Lab[:,:,0]
    a = new_src_img_Lab[:,:,1]
    b = new_src_img_Lab[:,:,2]
    print(l.shape)    
    l_tar = new_tar_img_Lab[:,:,0]
    a_tar = new_tar_img_Lab[:,:,1]
    b_tar = new_tar_img_Lab[:,:,2]
   
    lmean = np.mean(l)
    amean = np.mean(a)
    bmean = np.mean(b)
    print(lmean.shape)

    tar_lmean = np.mean(l_tar)
    tar_amean = np.mean(a_tar)
    tar_bmean = np.mean(b_tar)

    src_l_std = np.std(l)
    src_a_std = np.std(a)
    src_b_std = np.std(b)

    tar_l_std = np.std(l_tar)
    tar_a_std = np.std(a_tar)
    tar_b_std = np.std(b_tar)
    

###########################################
    lstar_src = l - lmean 
    astar_src = a - amean 
    bstar_src = b - bmean 

    #labstar = new_src_img_Lab - np.mean(new_src_img_Lab,axis=(0,1))
    #src_std =  np.std(new_src_img_Lab, axis=(0,1))
    #target_std =  np.std(new_tar_img_Lab, axis=(0,1))

    ldash1 = (tar_l_std/src_l_std) * lstar_src
    adash1 = (tar_a_std/src_a_std) * astar_src
    bdash1 = (tar_b_std/src_b_std) * bstar_src

    #labdash = target_std/src_std *labstar
    #mean_target = np.mean(new_tar_img_Lab, axis = (0,1))
    
    
    ldash_new = ldash1 + tar_lmean
    adash_new = adash1 + tar_amean
    bdash_new = bdash1 + tar_bmean


##########################################################

    combined_LAB[:,:,0] = ldash_new
    combined_LAB[:,:,1] = adash_new
    combined_LAB[:,:,2] = bdash_new
  
  
    final_rgb_img = convert_color_space_Lab_to_RGB(combined_LAB)

    final_img = convert_color_space_RGB_to_BGR(final_rgb_img)
    return (final_img)


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
    path_file_image_result_in_Lab = sys.argv[3]
 #   path_file_image_result_in_RGB = sys.argv[4]
 #   path_file_image_result_in_CIECAM97s = sys.argv[5]

    # ===== read input images
    img_RGB_source = cv2.imread(path_file_image_source).astype(np.float32) #is the image you want to change the its color
    img_RGB_target = cv2.imread(path_file_image_target).astype(np.float32) #is the image containing the color distribution that you want to change the img_RGB_source to (transfer color of the img_RGB_target to the img_RGB_source)
    

    rimage = cv2.imread(path_file_image_result_in_Lab).astype(np.float32)
    



    img_BGR = color_transfer(img_RGB_source, img_RGB_target, option='in_Lab')
    img_BGR = np.uint8(np.clip(img_BGR,0,255))
    print(np.max(img_BGR))
    print("RMSE:", np.sqrt((img_BGR - rimage)**2).mean())
    cv2.imwrite('path_file_image_result_in_Lab.png', img_BGR)
    # todo: save image to path_file_image_result_in_Lab

    #img_RGB_new_RGB       = color_transfer(img_RGB_source, img_RGB_target, option='in_RGB')
    # todo: save image to path_file_image_result_in_RGB

    #img_RGB_new_CIECAM97s = color_transfer(img_RGB_source, img_RGB_target, option='in_CIECAM97s')
    # todo: save image to path_file_image_result_in_CIECAM97s

