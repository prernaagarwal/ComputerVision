import cv2 
import sys
import numpy as np
from random import randint
import matplotlib.pyplot as plt

def ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000):
    '''
    Apply RANSAC algorithm to find a homography transformation matrix that align 2 sets of feature points, transform the first set of feature point to the second (e.g. warp image 1 to image 2)
    :param list_pairs_matched_keypoints: has the format as a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]],....]
    :param threshold_ratio_inliers: threshold on the ratio of inliers over the total number of samples, accept the estimated homography if ratio is higher than the threshold
    :param threshold_reprojtion_error: threshold of reprojection error (measured as euclidean distance, in pixels) to determine whether a sample is inlier or outlier
    :param max_num_trial: the maximum number of trials to do take sample and do testing to find the best homography matrix
    :return best_H: the best found homography matrix
    '''
    best_H = None

    print("len list_pairs_matched_keyponts: ",len(list_pairs_matched_keypoints))
    for pair in list_pairs_matched_keypoints:
        pt1 = pair[0]
        pt1.append(1)
        pt2 = pair[1]
        pt2.append(1)
    
    #### 1000 times

    max_inliers = 0
    listH = []
    for num in range(0,max_num_trial):

        A = []
        points4 = []
        otherH= []
        #Step 1: Randomly select a seed group of matches
        for i in range(4):
            index = randint(0,len(list_pairs_matched_keypoints)-1)
            points4.append(list_pairs_matched_keypoints[index])
            x = points4[i][0][0]
            y = points4[i][0][1]
            u = points4[i][1][0]
            v = points4[i][1][1]
            
            A.append([ x, y, 1, 0, 0, 0, -u*x, -u*y, -u ])
            A.append([ 0, 0, 0, x, y, 1, -v*x, -v*y, -v ])
        A = np.asarray(A)
        #AH=0
        #Step 2: Compute transformation from seed group
        U,S,V = np.linalg.svd(A)
        last = V[-1,:]
        H = last.reshape(3,3)
      
        

        #Step 3: Find inliers to this transformation 
        inliers = []
        for point_pair in list_pairs_matched_keypoints:
            p1 = np.asarray(point_pair[0])
            p2 = np.asarray(point_pair[1])
            projected_p2 = np.dot(H, p1) 
            projected_p2 /=projected_p2[2]
            #print(projected_p2)
            error = np.linalg.norm(projected_p2-p2)
            if (error < threshold_reprojtion_error):
                inliers.append(point_pair) 

        otherH.append(H)
        otherH.append(len(inliers)/len(list_pairs_matched_keypoints))
        listH.append(otherH)
        #print("H:", H)


       
        #Step4: If the number of inliers is sufficiently large, re-compute leastsquares estimate of transformation on all of the inliers
        if (len(inliers) / len(list_pairs_matched_keypoints) > threshold_ratio_inliers):
            
            if (len(inliers) > max_inliers):
                #print("len inliers:",len(inliers))
                #print("max_inliers:",max_inliers)
                max_inliers = len(inliers)
                A = []
                for point in inliers:
                    x = point[0][0]
                    y = point[0][1]
                    u = point[1][0]
                    v = point[1][1]

                    A.append([ x, y, 1, 0, 0, 0, -u*x, -u*y, -u ])
                    A.append([ 0, 0, 0, x, y, 1, -v*x, -v*y, -v ])
                A = np.asarray(A)
                #print(A.shape)
                #AH=0
                U,S,V = np.linalg.svd(A)
                last = V[-1,:]
                newH = last.reshape(3,3)
                #print("newH:",newH) 
           
                best_H = newH
                print("bestH:",best_H)
                if(best_H is None):
                    print("bestH is none")
                
        #print("########################################")

    if (best_H is None):
        listH.sort(key = lambda x: x[1])
        print("best_H not found in 1000 trials")
        best_H = listH[0][0]

    print("bestH:",best_H)
    return best_H

def ex_extract_and_match_feature(img_1, img_2, ratio_robustness=0.7):
    '''
    1/ extract SIFT feature from image 1 and image 2,
    2/ use a bruteforce search to find pairs of matched features: for each feature point in img_1, find its best matched feature point in img_2
    3/ apply ratio test to select the set of robust matched points
    :param img_1: input image 1
    :param img_2: input image 2
    :param ratio_robustness: ratio for the robustness test
    :return list_pairs_matched_keypoints: has the format as list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]]]
    '''
    # ==============================
    # ===== 1/ extract features from input image 1 and image 2
    # ==============================
    """ 
    sift = cv2.xfeatures2d.SIFT_create()
    kp1 = sift.detect(img_1,None)
    kpimg1=cv2.drawKeypoints(img_1,kp1,img_1)
    cv2.imwrite('kp_im1.jpg',kpimg1)
    kp1,des1 = sift.compute(img_1,kp1)
    pts1 = [kp1[i].pt for i in range(len(kp1))]
    
    sift = cv2.xfeatures2d.SIFT_create()
    kp2 = sift.detect(img_2,None)
    kpimg2=cv2.drawKeypoints(img_1,kp2,img_2)
    cv2.imwrite('kp_im2.jpg',kpimg2)
    kp2,des2 = sift.compute(img_2,kp2)
    pts2 = [kp2[i].pt for i in range(len(kp2))]
    """
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_1,None)
    pts1 = [kp1[i].pt for i in range(len(kp1))]
    sift = cv2.xfeatures2d.SIFT_create()
    kp2, des2 = sift.detectAndCompute(img_2,None)
    pts2 = [kp2[i].pt for i in range(len(kp2))]
    

    # ==============================
    # ===== 2/ use bruteforce search to find a list of pairs of matched feature points
    # ==============================
    list_pairs_matched_keypoints = []
    
    for i in range(len(des1)):
        error = []
        for j in range(len(des2)):
            point = []
            point.append(list(pts2[j]))
            val = np.linalg.norm(des1[i]-des2[j])
            point.append(val)
            error.append(point)
        error.sort(key = lambda distance: distance[1])
        if (error[0][1]/error[1][1] < 0.7):
            matched_points = []
            matched_points.append(list(pts1[i]))
            matched_points.append(error[0][0])
            list_pairs_matched_keypoints.append(matched_points)
    #print(list_pairs_matched_keypoints) #77

    return list_pairs_matched_keypoints



def ex_warp_blend_crop_image(img_1,H_1,img_2):
    '''
    1/ warp image img_1 using the homography H_1 to align it with image img_2 (using backward warping and bilinear resampling)
    2/ stitch image img_1 to image img_2 and apply average blending to blend the 2 images into a single panorama image
    3/ find the best bounding box for the resulting stitched image
    :param img_1:
    :param H_1:
    :param img_2:
    :return img_panorama: resulting panorama image
    '''
    img_panorama = None
    # =====  use a backward warping algorithm to warp the source
    # 1/ to do so, we first create the inverse transform; 2/ use bilinear interpolation for resampling
    # to be completed ...

    # ===== blend images: average blending
    # to be completed ...

    # ===== find the best bounding box for the resulting stitched image so that it will contain all pixels from 2 original images
    # to be completed ...

    return img_panorama

def stitch_images(img_1, img_2):
    '''
    :param img_1: input image 1. We warp this image to align and stich it to the image 2
    :param img_2: is the reference image. We will not warp this image
    :return img_panorama: the resulting stiched image
    '''
    print('==============================')
    print('===== stitch two images to generate one panorama image')
    print('==============================')

    # ===== extract and match features from image 1 and image 2
    list_pairs_matched_keypoints = ex_extract_and_match_feature(img_1=img_1, img_2=img_2, ratio_robustness=0.7)

    # ===== use RANSAC algorithm to find homography to warp image 1 to align it to image 2
    H_1 = ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojtion_error=3, max_num_trial=1000)


#    dst = cv2.warpPerspective(img_1,H_1, (300,300))
#    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    
    # ===== warp image 1, blend it with image 2 using average blending to produce the resulting panorama image
    img_panorama = ex_warp_blend_crop_image(img_1=img_1,H_1=H_1, img_2=img_2)

    return dst
    #return img_panorama

if __name__ == "__main__":
    print('==================================================')
    print('PSU CS 410/510, Winter 2019, HW2: image stitching')
    print('==================================================')

    path_file_image_1 = sys.argv[1]
    path_file_image_2 = sys.argv[2]
    path_file_image_result = sys.argv[3]


    # ===== read 2 input images
    img_1 = cv2.imread(path_file_image_1)
    img_2 = cv2.imread(path_file_image_2)
    
    #151,202,3
    #print("img_1 shape:", img_1.shape)
    #print("img_2 shape:", img_2.shape)
    # ===== create a panorama image by stitch image 1 to image 2
    img_panorama = stitch_images(img_1=img_1, img_2=img_2)

    # ===== save panorama image
    cv2.imwrite(filename=path_file_image_result, img=(img_panorama).clip(0.0, 255.0).astype(np.uint8))

