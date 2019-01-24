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
            LMSm = np.matmul(constm1, RGBm)     #get LSM color from equation((4)
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
    
    row,col,depth = img_RGB.shape               #row = 599, col = 599, depth = 3
    for i in range(row):
        for j in range(col):
            Labm  = img_Lab[i,j]                #each [i,j] has LAB values
            LMSm = np.matmul(constm3, Labm)     #get LSM from equation(8)
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

    row,col,depth = img_RGB.shape
    for i in range(row):
        for j in range(col):
            RGBm = img_RGB[i,j]
            LMSm = np.matmul(constm1, RGBm)

            CAM = np.matmul(constm2,LMSm)
            img_CIECAM97s[i,j] = CAM


    return img_CIECAM97s

def convert_color_space_CIECAM97s_to_RGB(img_CIECAM97s):
    '''
    convert image color space CIECAM97s to RGB
    '''
    img_RGB = np.zeros_like(img_CIECAM97s,dtype=np.float32)


    constm1 = [[1,1,1],[1,1,-1],[1,-2,0]]
    constm2 = [[np.sqrt(3)/3.0,0,0],[0,np.sqrt(6)/6.0,0],[0,0,np.sqrt(2)/2.0]]
    constm3 = np.matmul(constm1,constm2)
    constm4 = [[4.4679, -3.5873, 0.1193],[-1.2186,2.3809,-0.1624],[0.0497, -0.2439, 1.2045]]
    
    row,col,depth = img_CIECAM97s.shape
    for i in range(row):
        for j in range(col):
            CAMm  = img_CIECAM97s[i,j]
            LMSm = np.matmul(constm3, CAMm)

            RGB = np.matmul(constm4,LMSm)
            img_RGB[i,j] = RGB

    return img_RGB


def color_transfer_in_Lab(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_Lab =====')

    #convert BGR source and target image to RGB
    new_rgb_img = convert_color_space_BGR_to_RGB(img_RGB_source)
    target_rgb = convert_color_space_BGR_to_RGB(img_RGB_target)
    
    #convert from RGB colorspace to LAB colorspace
    new_src_img_Lab = convert_color_space_RGB_to_Lab(new_rgb_img)
    new_tar_img_Lab = convert_color_space_RGB_to_Lab(target_rgb) 
    
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
    return (final_img)




def color_transfer_in_RGB(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_RGB =====')
    # to be completed ...



def color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_CIECAM97s =====')
    
    new_rgb_img = convert_color_space_BGR_to_RGB(img_RGB_source)
    target_rgb = convert_color_space_BGR_to_RGB(img_RGB_target)

    img_src = convert_color_space_RGB_to_CIECAM97s(new_rgb_img)
    img_tar = convert_color_space_RGB_to_CIECAM97s(target_rgb)



    combined_CAM = np.zeros_like(img_src,dtype=np.float32)

    l = img_src[:,:,0]
    a = img_src[:,:,1]
    b = img_src[:,:,2]

    l_tar = img_tar[:,:,0]
    a_tar = img_tar[:,:,1]
    b_tar = img_tar[:,:,2]
   
    lmean = np.mean(l)
    amean = np.mean(a)
    bmean = np.mean(b)

    tar_lmean = np.mean(l_tar)
    tar_amean = np.mean(a_tar)
    tar_bmean = np.mean(b_tar)

    src_l_std = np.std(l)
    src_a_std = np.std(a)
    src_b_std = np.std(b)

    tar_l_std = np.std(l_tar)
    tar_a_std = np.std(a_tar)
    tar_b_std = np.std(b_tar)
    

    lstar_src = l - lmean 
    astar_src = a - amean 
    bstar_src = b - bmean 

    ldash1 = (tar_l_std/src_l_std) * lstar_src
    adash1 = (tar_a_std/src_a_std) * astar_src
    bdash1 = (tar_b_std/src_b_std) * bstar_src

    ldash_new = ldash1 + tar_lmean
    adash_new = adash1 + tar_amean
    bdash_new = bdash1 + tar_bmean

    combined_CAM[:,:,0] = ldash_new
    combined_CAM[:,:,1] = adash_new
    combined_CAM[:,:,2] = bdash_new
  

    img_src_final = convert_color_space_CIECAM97s_to_RGB(combined_CAM)
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
 #   path_file_image_result_in_RGB = sys.argv[4]        #store my RGB result
 #   path_file_image_result_in_CIECAM97s = sys.argv[5]  #store my CIECAM97s result


    # ===== read input images
    img_RGB_source = cv2.imread(path_file_image_source).astype(np.float32) #is the image you want to change the its color
    img_RGB_target = cv2.imread(path_file_image_target).astype(np.float32) #is the image containing the color distribution that you want to change the img_RGB_source to (transfer color of the img_RGB_target to the img_RGB_source)
    


    rimage = cv2.imread(path_file_image_result_in_Lab).astype(np.float32)
    



    img_BGR = color_transfer(img_RGB_source, img_RGB_target, option='in_Lab')
    #img_BGR = color_transfer(img_RGB_source, img_RGB_target, option='in_CIECAM97s')
    img_BGR = np.uint8(np.clip(img_BGR,0,255))

    #find max value and RMSE error between my result and given result
    print(np.max(img_BGR))
    print("RMSE:", np.sqrt((img_BGR - rimage)**2).mean())
    
    # todo: save image to path_file_image_result_in_Lab
    cv2.imwrite('path_file_image_result_in_Lab.png', img_BGR)



    #img_RGB_new_RGB       = color_transfer(img_RGB_source, img_RGB_target, option='in_RGB')
    # todo: save image to path_file_image_result_in_RGB



    #img_RGB_new_CIECAM97s = color_transfer(img_RGB_source, img_RGB_target, option='in_CIECAM97s')
    # todo: save image to path_file_image_result_in_CIECAM97s

