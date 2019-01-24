# Prerna Agarwal
# Computer Vision CS410
# HW1
# Based on Color Transfer between Images research paper
# Kindly refer to the equations from the research paper
# mentioned above.

# Note: This program takes some time to run because of nested for loops

import cv2
import numpy as np
import sys

def convert_color_space_BGR_to_RGB(img_BGR):
    img_RGB = np.zeros_like(img_BGR,dtype=np.float32)
    #manually converting BGR to RGB
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
    #manually converting RGB to BGR
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
    img_Lab = np.zeros_like(img_RGB,dtype=np.float32)
   
   
    #matrix from equation(4)
    constm1 = np.array([[0.3811,0.5783,0.0402],[0.1967,0.7244,0.0782],[0.0241,0.1288,0.8444]])
    #matrix from equation(6)
    constm2 = [[1/np.sqrt(3),0,0],[0,1/np.sqrt(6),0],[0,0,1/np.sqrt(2)]]
    #matrix from equation(6)
    constm3 = [[1,1,1],[1,1,-2],[1,-1,0]]
    constm4 = np.matmul(constm2,constm3)

     
    row,col,depth = img_RGB.shape               #row = 599, col = 800, depth = 3
    for i in range(row):
        for j in range(col):
            RGBm = img_RGB[i,j]                 #each [i,j] has RGB values
            LMSm = np.matmul(constm1, RGBm)     #get LMS color from equation((4)
            LMSlog = np.log10(LMSm)             #convert to logarithmic space to remove skew

            LAB = np.matmul(constm4,LMSlog)     #get Lab from equation(6)
            img_Lab[i,j] = LAB                  #store that Lab value for the RGB value at [i,j]

    #print(img_Lab)
    return img_Lab

def convert_color_space_Lab_to_RGB(img_Lab):
    '''
    convert image color space Lab to RGB
    '''
    img_LMS = np.zeros_like(img_Lab,dtype=np.float32)
    img_RGB = np.zeros_like(img_Lab,dtype=np.float32)
    
    #matrix from equation(8)
    constm1 = [[1,1,1],[1,1,-1],[1,-2,0]]
    #matrix from equation(8)
    constm2 = [[np.sqrt(3)/3.0,0,0],[0,np.sqrt(6)/6.0,0],[0,0,np.sqrt(2)/2.0]]
    constm3 = np.matmul(constm1,constm2)
    #matrix from equation(9)
    constm4 = [[4.4679, -3.5873, 0.1193],[-1.2186,2.3809,-0.1624],[0.0497, -0.2439, 1.2045]]
    
    row,col,depth = img_RGB.shape               #row = 599, col = 800, depth = 3
    for i in range(row):
        for j in range(col):
            Labm  = img_Lab[i,j]                #each [i,j] has LAB values
            LMSm = np.matmul(constm3, Labm)     #get LMS from equation(8)
            LMS10 = np.power(10,LMSm)           #convert to linear space

            RGB = np.matmul(constm4,LMS10)      #get RGB from equation(9)
            img_RGB[i,j] = RGB                  #store that RGB value for the Lab value at [i,j]

    return img_RGB

def convert_color_space_RGB_to_CIECAM97s(img_RGB):
    '''
    convert image color space RGB to CIECAM97s
    '''
    img_CIECAM97s = np.zeros_like(img_RGB,dtype=np.float32)

    constm1 = np.array([[0.3811,0.5783,0.0402],[0.1967,0.7244,0.0782],[0.0241,0.1288,0.8444]])
    constm2 = [[2.00,1.00,0.05],[1.00,-1.09,0.09],[0.11,0.11,-0.22]]

    row,col,depth = img_RGB.shape               #row = 599, col = 800, depth = 3
    for i in range(row):
        for j in range(col):
            RGBm = img_RGB[i,j]                 #each [i,j] has RGB values
            LMSm = np.matmul(constm1, RGBm)     #get LMS from equation(4)

            CAM = np.matmul(constm2,LMSm)       #get CIECAM97s from equation on page 39
            img_CIECAM97s[i,j] = CAM            #store that CIECAM97s value for the RGB value at [i,j]


    return img_CIECAM97s

def convert_color_space_CIECAM97s_to_RGB(img_CIECAM97s):
    '''
    convert image color space CIECAM97s to RGB
    '''
    img_RGB = np.zeros_like(img_CIECAM97s,dtype=np.float32)


    constm1 = [[2.00,1.00,0.05],[1.00,-1.09,0.09],[0.11,0.11,-0.22]]
    constm2 = np.linalg.inv(constm1)            #inverse of constm1
    constm3 = [[4.4679, -3.5873, 0.1193],[-1.2186,2.3809,-0.1624],[0.0497, -0.2439, 1.2045]]
    
    row,col,depth = img_CIECAM97s.shape         #row = 599, col = 800, depth = 3
    for i in range(row):    
        for j in range(col):    
            CAMm  = img_CIECAM97s[i,j]          #each [i,j] has CIECAM97s values
            LMSm = np.matmul(constm2, CAMm)     #get LMS by taking inverse of equation on page 39

            RGB = np.matmul(constm3,LMSm)       #get RGB from equation 9
            img_RGB[i,j] = RGB                  #store that RGB value for the CIECAM97s value at [i,j]

    return img_RGB


def color_transfer_in_Lab(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_Lab =====')

    #convert BGR source and target image to RGB
    new_rgb_src = convert_color_space_BGR_to_RGB(img_RGB_source)
    new_rgb_tar = convert_color_space_BGR_to_RGB(img_RGB_target)
    
    #convert from RGB colorspace to LAB colorspace
    new_src_img_Lab = convert_color_space_RGB_to_Lab(new_rgb_src)
    new_tar_img_Lab = convert_color_space_RGB_to_Lab(new_rgb_tar) 
    
    combined_LAB = np.zeros_like(new_src_img_Lab,dtype=np.float32)

    #statistics and color correction
    #separate out l, a and b for source and target images respectively
    l = new_src_img_Lab[:,:,0]
    a = new_src_img_Lab[:,:,1]
    b = new_src_img_Lab[:,:,2]

    l_tar = new_tar_img_Lab[:,:,0]
    a_tar = new_tar_img_Lab[:,:,1]
    b_tar = new_tar_img_Lab[:,:,2]
   
    #compute mean of respective source and target l,a,b
    lmean = np.mean(l)
    amean = np.mean(a)
    bmean = np.mean(b)

    tar_lmean = np.mean(l_tar)
    tar_amean = np.mean(a_tar)
    tar_bmean = np.mean(b_tar)

    #compute std dev. of respective source and target l,a,b
    src_l_std = np.std(l)
    src_a_std = np.std(a)
    src_b_std = np.std(b)

    tar_l_std = np.std(l_tar)
    tar_a_std = np.std(a_tar)
    tar_b_std = np.std(b_tar)
    
    #subtract the mean from data points
    lstar_src = l - lmean 
    astar_src = a - amean 
    bstar_src = b - bmean 

    #scale the data points of synthetic image(src) by factors determined by
    #the respective stf. dev.
    ldash1 = (tar_l_std/src_l_std) * lstar_src
    adash1 = (tar_a_std/src_a_std) * astar_src
    bdash1 = (tar_b_std/src_b_std) * bstar_src

    #add the mean of photograph(target) 
    ldash_new = ldash1 + tar_lmean
    adash_new = adash1 + tar_amean
    bdash_new = bdash1 + tar_bmean

    #stack l, a and b in a matrix
    combined_LAB[:,:,0] = ldash_new
    combined_LAB[:,:,1] = adash_new
    combined_LAB[:,:,2] = bdash_new
  
    #convert from Lab colorspace to RGB colorspace
    final_rgb_img = convert_color_space_Lab_to_RGB(combined_LAB)
    #convert from RGB to BGR
    final_img = convert_color_space_RGB_to_BGR(final_rgb_img)
    return final_img




def color_transfer_in_RGB(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_RGB =====')

    #convert BGR source and target image to RGB
    new_src_img = convert_color_space_BGR_to_RGB(img_RGB_source)
    new_tar_img = convert_color_space_BGR_to_RGB(img_RGB_target)


    combined_RGB = np.zeros_like(new_src_img,dtype=np.float32)

    #statistics and color correction
    #separate out r, g and b for source and target images respectively
    r = new_src_img[:,:,0]
    g = new_src_img[:,:,1]
    b = new_src_img[:,:,2]

    r_tar = new_tar_img[:,:,0]
    g_tar = new_tar_img[:,:,1]
    b_tar = new_tar_img[:,:,2]
   
    #compute mean of respective source and target r,g,b
    rmean = np.mean(r)
    gmean = np.mean(g)
    bmean = np.mean(b)

    tar_rmean = np.mean(r_tar)
    tar_gmean = np.mean(g_tar)
    tar_bmean = np.mean(b_tar)

    #compute std dev. of respective source and target r,g,b
    src_r_std = np.std(r)
    src_g_std = np.std(g)
    src_b_std = np.std(b)

    tar_r_std = np.std(r_tar)
    tar_g_std = np.std(g_tar)
    tar_b_std = np.std(b_tar)
    
    #subtract the mean from data points
    rstar_src = r - rmean 
    gstar_src = g - gmean 
    bstar_src = b - bmean 

    #scale the data points of synthetic image(src) by factors determined by
    #the respective stf. dev.
    rdash1 = (tar_r_std/src_r_std) * rstar_src
    gdash1 = (tar_g_std/src_g_std) * gstar_src
    bdash1 = (tar_b_std/src_b_std) * bstar_src

    #add the mean of photograph(target) 
    rdash_new = rdash1 + tar_rmean
    gdash_new = gdash1 + tar_gmean
    bdash_new = bdash1 + tar_bmean

    #stack l, a and b in a matrix
    combined_RGB[:,:,0] = rdash_new
    combined_RGB[:,:,1] = gdash_new
    combined_RGB[:,:,2] = bdash_new


    #convert from RGB to BGR
    final_img = convert_color_space_RGB_to_BGR(combined_RGB)
    return final_img





def color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_CIECAM97s =====')
    
    #convert BGR source and target image to RGB
    new_rgb_src = convert_color_space_BGR_to_RGB(img_RGB_source)
    new_rgb_tar = convert_color_space_BGR_to_RGB(img_RGB_target)

    #convert from BGR colorspace to CIECAM97s colorspace
    img_src = convert_color_space_RGB_to_CIECAM97s(new_rgb_src)
    img_tar = convert_color_space_RGB_to_CIECAM97s(new_rgb_tar)

    combined_CAM = np.zeros_like(img_src,dtype=np.float32)

    #color correction based on mean and std. dev
    a = img_src[:,:,0]
    c1 = img_src[:,:,1]
    c2 = img_src[:,:,2]

    a_tar = img_tar[:,:,0]
    c1_tar = img_tar[:,:,1]
    c2_tar = img_tar[:,:,2]
   
    #compute mean of respective source and target a,c1,c2
    amean = np.mean(a)
    c1mean = np.mean(c1)
    c2mean = np.mean(c2)

    tar_amean = np.mean(a_tar)
    tar_c1mean = np.mean(c1_tar)
    tar_c2mean = np.mean(c2_tar)

    #compute std dev. of respective source and target a,c1,c2
    src_a_std = np.std(a)
    src_c1_std = np.std(c1)
    src_c2_std = np.std(c2)

    tar_a_std = np.std(a_tar)
    tar_c1_std = np.std(c1_tar)
    tar_c2_std = np.std(c2_tar)
    

    astar_src = a - amean 
    c1star_src = c1 - c1mean 
    c2star_src = c2 - c2mean 

    #scale the data points of synthetic image(src) by factors determined by
    #the respective stf. dev.
    adash1 = (tar_a_std/src_a_std) * astar_src
    c1dash1 = (tar_c1_std/src_c1_std) * c1star_src
    c2dash1 = (tar_c2_std/src_c2_std) * c2star_src

    adash_new = adash1 + tar_amean
    c1dash_new = c1dash1 + tar_c1mean
    c2dash_new = c2dash1 + tar_c2mean

    combined_CAM[:,:,0] = adash_new
    combined_CAM[:,:,1] = c1dash_new
    combined_CAM[:,:,2] = c2dash_new
  

    #convert from CIECAM97s colorspace to RGB
    img_src_final = convert_color_space_CIECAM97s_to_RGB(combined_CAM)
    
    #convert from RGB to BGR
    final_img = convert_color_space_RGB_to_BGR(img_src_final)
    
    return final_img



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

    path_file_image_source = sys.argv[1]                #variable to store source image
    path_file_image_target = sys.argv[2]                #target image
    path_file_image_result_in_Lab = sys.argv[3]         #store my Lab result
    path_file_image_result_in_RGB = sys.argv[4]         #store my RGB result
    path_file_image_result_in_CIECAM97s = sys.argv[5]   #store my CIECAM97s result

    path_file_image_given_result = sys.argv[6]          #store the given result

    # ===== read input images =====#
    img_RGB_source = cv2.imread(path_file_image_source).astype(np.float32) #is the image you want to change the its color
    img_RGB_target = cv2.imread(path_file_image_target).astype(np.float32) #is the image containing the color distribution that you want to change the img_RGB_source to (transfer color of the img_RGB_target to the img_RGB_source)
    
    # ==== read given result image ====#
    result_image = cv2.imread(path_file_image_given_result).astype(np.float32)
    

    ############### LAB #############################
    img_RGB_new_Lab = color_transfer(img_RGB_source, img_RGB_target, option='in_Lab')
    img_RGB_new_Lab = np.uint8(np.clip(img_RGB_new_Lab,0,255))

    #find max value and RMSE error between my result and given result
    #print(np.max(img_RGB_new_Lab))
    print("RMSE:", np.sqrt((img_RGB_new_Lab - result_image)**2).mean())
    
    # todo: save image to path_file_image_result_in_Lab
    cv2.imwrite(path_file_image_result_in_Lab, img_RGB_new_Lab)
    ################################################
    

    ############## RGB #############################
    img_RGB_new_RGB = color_transfer(img_RGB_source, img_RGB_target, option='in_RGB')
    img_RGB_new_RGB = np.uint8(np.clip(img_RGB_new_RGB,0,255))

    # todo: save image to path_file_image_result_in_RGB
    cv2.imwrite(path_file_image_result_in_RGB, img_RGB_new_RGB)
    ###############################################

    ############## CIECAM97s #######################    
    img_RGB_new_CIECAM97s = color_transfer(img_RGB_source, img_RGB_target, option='in_CIECAM97s')
    img_RGB_new_CIECAM97s = np.uint8(np.clip(img_RGB_new_CIECAM97s,0,255))

    # todo: save image to path_file_image_result_in_CIECAM97s
    cv2.imwrite(path_file_image_result_in_CIECAM97s, img_RGB_new_CIECAM97s)
    ###############################################



