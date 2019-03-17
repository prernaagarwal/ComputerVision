#!/usr/bin/python3
# Prerna Agarwal
# SLIC Project
# For this assignment, I followed jayrambhia tutorial

import numpy as np
import sys
import cv2
import math

class SLIC:

    #function to initialize the class data members
    def __init__(self, img, K):
        self.img = img
        #get height and width of the image
        self.h, self.w = img.shape[:2]
        #convert bgr to lab colorspace
        self.labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
        self.K = K #Number of clusters
        self.N = img.shape[0]*img.shape[1] # Number of pixels
        self.S = int(math.sqrt(self.N/self.K)) # distance between centers
        self.m = 10  #recommended by the paper, value can be between 1-20
        self.MAX = np.inf
        self.iterations = 15
        #inititialize the centers of the clusters
        self.centers = self.initialize_centers()
        #initialize the clusters matrix
        self.clusters = -1 * np.ones((self.h, self.w))
        #initialize the distane matrix
        self.distances = self.MAX * np.ones((self.h, self.w))



    # function to initialize the centers of the clusters
    # self.S = the distance between the centers of the clusters
    # loop through the entire image looking for centers
    # self.S distance apart with the lowest gradient value in each
    # 3x3 neighborhood
    def initialize_centers(self):

        #list of senters
        centerList = []

        i = self.S
        while i < (self.w - self.S//2):
            j = self.S
            while j < (self.h - self.S//2):
               
                x,y = self.getCenter([i, j]) #get x,y
                l,a,b = self.labimg[y, x]     #get l,a,b
                center = [l, a, b, x, y]
                centerList.append(center)
                j += self.S

            i += self.S

        return np.array(centerList)



    # function to get lowest gradient center in a 3x3 neighborhood
    # avoid choosing noisy pixel for center and also avoid the edges
    def getCenter(self, center):
        gradient = self.MAX
        minxy = center
      
        #3x3 neighborhood
        start1 = center[0] - 1
        stop1 = center[0] + 2
        start2 = center[1] - 1
        stop2 = center[1] + 2

        for i in range(start1, stop1):
            for j in range(start2, stop2):

                val1 = self.labimg[j+1, i]
                val2 = self.labimg[j, i+1]
                val3 = self.labimg[j, i]
                
                #check if new gradient is lower than the current gradient
                if (math.sqrt(np.square(val1[0] - val3[0])) + math.sqrt(np.square(val2[0] - val3[0]))) < gradient:
                    gradient = abs(val1[0] - val3[0]) + abs(val2[0] - val3[0])
                    minxy = [i, j]

        return minxy

    #function to check if the values are out of bounds
    def checkbounds(self, x1, y1, x2, y2):

        x1 = 0 if x1 <= 0 else x1
        y1 = 0 if y1 <= 0 else y1
        x2 = self.w if x2 > self.w else x2
        y2 = self.h if y2 > self.h else y2

        return x1, y1, x2, y2


    #adaptation of kmeans algorithm
    def algorithm(self):

        mat1 = np.mgrid[0:self.h,0:self.w].swapaxes(0,2)
        mat1 = mat1.swapaxes(0,1)

        for i in range(self.iterations):

            self.distances = self.MAX * np.ones((self.img.shape[0],self.img.shape[1]))

            for num in range(len(self.centers)):
                
                # center = [l,a,b,x,y]
                          #[0,1,2,3,4]

                #get bounds of 2Sx2S area around the center
                x = self.centers[num][3]
                y = self.centers[num][4]
                x1 = int(x - self.S)
                x2 = int(x + self.S)
                y1 = int(y - self.S)
                y2 = int(y + self.S)

                #check the 2Sx2S coordinates are not of out of bounds
                x1, y1, x2, y2 = self.checkbounds(x1,y1,x2,y2)

                #calculate lab euclidean distance
                dist_lab = np.sqrt(np.sum(np.square(self.labimg[y1:y2,x1:x2] - self.labimg[int(y),int(x)]), axis = 2))

                #calculate xy euclidean distance
                gridy, gridx = np.ogrid[y1 : y2, x1 : x2]
                dist_xy =  np.sqrt(np.square(gridy - y) + np.square(gridx - x))

                #weighted distance
                dist = np.sqrt(np.square(dist_lab / self.m) + np.square(dist_xy / self.S))

                #update the distances
                newDistance = self.distances[y1:y2, x1:x2]
                val = dist < newDistance
                newDistance[val] = dist[val]
                self.distances[y1:y2, x1:x2] = newDistance
                self.clusters[y1:y2, x1:x2][val] = num
        
            #update the cluster centers
            for c in range(len(self.centers)):
                val = (self.clusters == c)

                #update lab
                colr = self.labimg[val]
                dist1 = mat1[val]
                self.centers[c][0:3] = np.sum(colr, axis=0)
                #update xy
                sumy, sumx = np.sum(dist1, axis = 0)
                self.centers[c][3:] = sumx, sumy
                self.centers[c] /= np.sum(val)



    #function to display the borders of superpixels.
    def displaySuperpixels(self, color):
        X = [-1, -1, 0, 1, 1, 1, 0, -1]
        Y = [0, -1, -1, -1, 0, 1, 1, 1]

        val = np.zeros(self.img.shape[:2], np.bool)
        border = []

        for i in range(self.w):
            for j in range(self.h):
                num = 0
                for x1, y1 in zip(X, Y):
                    x = i + x1
                    y = j + y1
                    #check within bounds
                    if x>=0 and x < self.w and y>=0 and y < self.h:
                        if val[y, x] == False and self.clusters[j, i] != self.clusters[y, x]:
                            num += 1

                if num >= 2:
                    val[j, i] = True
                    border.append([j, i])

        for i in range(len(border)):
            self.img[border[i][0], border[i][1]] = color


if __name__ == "__main__":
    print("===================================================")
    print("Simple Linear Iterative Clustering (SLIC)")
    print("===================================================")

    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    K = int(sys.argv[2]) # number of clusters
   
    s = SLIC(img, K)
    s.algorithm()
    s.displaySuperpixels(0) #black color
    cv2.imshow("superpixels", s.img)
    cv2.imwrite("result.jpg", s.img)

