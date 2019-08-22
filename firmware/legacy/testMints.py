
import numpy as np
import cv2
import glob
import argparse


imgpoints = [] # 2d points in image plane.

imgOrig = cv2.imread('DSC_0016.JPG')

# print(str(images_left[i]))
# img_r = cv2.imread(images_right[i])
# print(str(images_right[i]))

scale_percent = 20# percent of original size
width  = int(imgOrig.shape[1] * scale_percent / 100)
height = int(imgOrig.shape[0] * scale_percent / 100)
dim = (width, height)
Orig = cv2.resize(imgOrig, dim, interpolation = cv2.INTER_AREA)




gray = cv2.cvtColor(Orig, cv2.COLOR_BGR2GRAY)
# gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
#
print(Orig)



# resize image


cv2.imshow('img',gray)
cv2.waitKey(1000)
# print(gray_r)
# Find the chess board corners

ret, corners = cv2.findChessboardCorners(gray,(6,7),None)
# ret_r, corners_r = cv2.findChessboardCorners(gray_r, (10, 7), None)
print(ret)
imgpoints.append(corners)
# Draw and display the corners
imgCor = cv2.drawChessboardCorners(Orig, (6,7), corners,ret)




print(imgCor)
cv2.imshow('img',imgCor)
cv2.waitKey(0)
# print(ret_l)
# # print(ret_l)
# print(corners_l)
# # print(corners_r)



def resizeImage(imgOrig,scalePercentage):
    width  = int(imgOrig.shape[1] * scalePercent / 100)
    height = int(imgOrig.shape[0] * scalePercent / 100)
    dim = (width, height)
    Orig = cv2.resize(imgOrig, dim, interpolation = cv2.INTER_AREA)
    return(Orig)
