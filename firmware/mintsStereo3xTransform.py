import numpy as np
import cv2
import glob
import argparse
import time

import os
import copy

import pickle


# Create the haar cascade
# faceCascade = cv2.CascadeClassifier(cascPath)

def readArgs():

    import argparse
    import os
    from distutils.util import strtobool
    import numbers

    #set up command line arguments
    parser = argparse.ArgumentParser(description="-- Stereo MINTS --")
    parser.add_argument('-f','--filePath', dest='filePath', help="Path to Image Files. (e.g. '-f ../img')")
    parser.add_argument('-x','--xSquares', dest='xSquares', help="# of horizontal squares")
    parser.add_argument('-y','--ySquares', dest='ySquares', help="# of vertical squares")
    args = parser.parse_args()

    #
    if args.filePath== None:
        print("Error: No file path given.")
        exit(1)
    #
    if args.xSquares== None:
        print("Error: No X square count given")
        exit(1)
    #
    if args.ySquares== None:
        print("Error: No Y square count given")
        exit(1)

    filePath   = str(args.filePath)
    xSquares    = int(args.xSquares)
    ySquares    = int(args.ySquares)

    argsOut = {\
                "filePath": filePath,\
                "xSquares":xSquares,\
                "ySquares":ySquares\
                }
    print(argsOut)
    return argsOut;

def invertGreyScale(imagem):
    imageNeg = cv2.bitwise_not(imagem)
    return imageNeg

def resizeImage(imgOrig,scalePercent):
    width  = int(imgOrig.shape[1] * scalePercent / 100)
    height = int(imgOrig.shape[0] * scalePercent / 100)
    dim = (width, height)
    Orig = cv2.resize(imgOrig, dim, interpolation = cv2.INTER_AREA)
    return(Orig);

def order_points(pts):

	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped


def getUndistortedImage(imagePath,pickleFileName,camera):

    imgPre =resizeImage(cv2.imread(imagePath),100)
    imgOrig = resizeImage(cv2.imread(imagePath),100)
    # Calibrating the Camera

    with open(pickleFileName, 'rb') as f:
        camData = pickle.load(f)

    print(camData)


    imgUndistorted  = cv2.undistort(imgPre, \
                        camData['cameraMatrix'+camera], \
                        camData['distortionCoefficients'+camera], \
                        None,\
                        camData['cameraMatrix'+camera])

    cv2.imshow(camera,imgOrig)
    cv2.waitKey(2000)

    cv2.imshow(camera+"Undistored",imgUndistorted)
    cv2.waitKey(2000)
    return imgUndistorted


def getCommonPoints(undistortedImage,camera,yPoints,xPoints):

    imgOrig = undistortedImage + 1 - 1
    # # Resize Image Here


    gray = cv2.cvtColor(undistortedImage, cv2.COLOR_BGR2GRAY)

    # grayPre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray    = cv2.equalizeHist(grayPre)
                        # Create our shapening kernel,


    ret, corners = cv2.findChessboardCorners(gray, (yPoints, xPoints), None)
    print(ret)
    if(camera is "Thermal"):
        corners = np.flip(corners, 0)
        print("Reversing")

    imgCor = cv2.drawChessboardCorners(imgOrig, (yPoints,xPoints), corners,ret)


    cv2.imshow(camera+"Undistored",imgCor)
    cv2.waitKey(2000)
    return ret, corners;

if __name__ == '__main__':

    print("Reading Arguments")
    parser = readArgs()

    print("Left Image")
    undistortedLeft =  getUndistortedImage(parser['filePath']+"/calibration/farAway/2019_08_21_15_46_20_left.jpg",\
                        "leftCamCalibration",\
                        "Left")

    validLeft,commonPointsLeft = getCommonPoints(undistortedLeft,\
                                    "Left",\
                                    parser['ySquares']-1,\
                                    parser['xSquares']-1)

    print("Right Image")
    undistortedRight =  getUndistortedImage(parser['filePath']+"/calibration/farAway/2019_08_21_15_46_20_right.jpg",\
                        "rightCamCalibration",\
                        "Right")

    validRight,commonPointsRight = getCommonPoints(undistortedRight,\
                                    "Right",\
                                    parser['ySquares']-1,\
                                    parser['xSquares']-1)


    print("Thermal Image")
    undistortedThermal =  getUndistortedImage(parser['filePath']+"/calibration/farAway/2019_08_21_15_46_20_thermal.jpg",\
                        "thermalCamCalibration",\
                        "Thermal")




    validThermal,commonPointsThermal = getCommonPoints(undistortedThermal,\
                                    "Thermal",\
                                    parser['ySquares']-1,\
                                    parser['xSquares']-1)


    print("======= Getting Homography Transformation ========")
    rows,cols,ch = undistortedLeft.shape
    homographyTM , __  = cv2.findHomography(commonPointsLeft, commonPointsThermal, cv2.RANSAC, 4)
    homographyImage = cv2.warpPerspective(undistortedLeft,homographyTM,(cols,rows))

    alpha = 0.4
    beta = (1.0 - alpha)
    mergedHomographyLocal = cv2.addWeighted(undistortedThermal,alpha,homographyImage, beta, 0.0)
    cv2.imshow("Homography" , mergedHomographyLocal)
    cv2.waitKey(10000)

    # cal_data = minstStereoTransform(parser['filePath'],\
    #                             parser['xSquares']-1,\
    #                             parser['ySquares']-1\
    # #                             )
    # print(cal_data.leftImagesAll[0])
    #





    # stereoCamData = {
	# "cameraMatrixLeft" :cal_data.M1,
	# "cameraMatrixRight" :cal_data.M2,
    # "distortionCoefficientsLeft" :cal_data.d1,
    # "distortionCoefficientsRight" :cal_data.d2,
    #      }
    #
    #
    # with open('stereoCamData.pickle', 'wb') as f:
    #     pickle.dump(stereoCamData, f)


    # # print("--------------------")
    #
    # print(cal_data.corners_l[0][0])
    # print(cal_data.corners_r[0][0])



    # print(cal_data.M1)
    # print(cal_data.d1)
    #
    # #
    # (leftRectification, rightRectification, leftProjection, rightProjection,
    #     dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
    #             cal_data.M1,cal_data.d1,
    #             cal_data.M2,cal_data.d2,
    #             cal_data.image_size,
    #             cal_data.rotational_matrix,
    #             cal_data.translational_matrix,
    #             None,
    #             None,
    #             None,
    #             None,
    #             None,
    #             cv2.CALIB_ZERO_DISPARITY, 0)
    #
    # leftMapX, leftMapY = cv2.initUndistortRectifyMap(
    #         cal_data.M1, cal_data.d1, leftRectification,
    #         leftProjection, cal_data.image_size, cv2.CV_32FC1)
    # rightMapX, rightMapY = cv2.initUndistortRectifyMap(
    #         cal_data.M2, cal_data.d2, rightRectification,
    #         rightProjection, cal_data.image_size, cv2.CV_32FC1)
    #
    #
    #
    # ## Get Images
    #
    # for i, fname in enumerate(cal_data.leftImagesAll):
    #
    #     fixedLeft = cv2.remap(cal_data.leftImagesAll[i], leftMapX, leftMapY,cv2.INTER_LINEAR)
    #     fixedRight = cv2.remap(cal_data.rightImagesAll[i], rightMapX, rightMapY,cv2.INTER_LINEAR)
    #
    #     cv2.imshow("Fixed Left", fixedLeft)
    #     cv2.waitKey(1000)
    #
    #     cv2.imshow("Fixed Right", fixedRight)
    #     cv2.waitKey(1000)
    #
    #
    #
    #
    # stereoMatcher = cv2.StereoBM_create()
    #
    # grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    # grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
    # depth = stereoMatcher.compute(grayLeft, grayRight)
