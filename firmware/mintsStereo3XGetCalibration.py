import numpy as np
import cv2
import glob
import argparse
import time

import os
import datetime
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

def resizeImage(self,imgOrig,scalePercent):
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


class minstThreeFoldCamParametors(object):

    def __init__(self, filepath,xPoints,yPoints):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((yPoints*xPoints, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:yPoints, 0:xPoints].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.

        self.objpointsLeft = []  # 3d point in real world space
        self.objpointsRight = []
        self.objpointsThermal = []

        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.
        self.imgpoints_t = []  # 2d points in image plane.

        self.yPoints = yPoints
        self.xPoints = xPoints
        # Additions by Lakitha
        # Have a Generic Size
        self.rotational_matrix = []
        self.translational_matrix = []

        self.img_size_l        = []
        self.img_size_r        = []
        self.img_size_t        = []

        self.leftImagesAll  = []
        self.rightImagesAll = []
        self.thermalImagesAll  = []

        self.leftImagesAllOrig  = []
        self.rightImagesAllOrig = []
        self.thermalImagesAllOrig = []

        self.corners_l = []
        self.corners_r = []
        self.corners_t = []

        self.cal_path = filepath

        self.readLeft(self.cal_path)
        self.readRight(self.cal_path)
        self.readThermal(self.cal_path)

    def readLeft(self, cal_path):

        print("Reading Left Images")
        images= glob.glob(cal_path + 'left/*.jpg')
        images.sort()


        for i, fname in enumerate(images):

            img = resizeImage(self,cv2.imread(images[i]),100)
            img_Orig = resizeImage(self,cv2.imread(images[i]),100)

            grayPre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grayPre2    = cv2.equalizeHist(grayPre)
                        # Create our shapening kernel, it must equal to one eventually
            kernel_sharpening = np.array([[-1,-1,-1],
                                          [-1, 9,-1],
                                          [-1,-1,-1]])
            # applying the sharpening kernel to the input image & displaying it.
            gray = cv2.filter2D(grayPre2, -1, kernel_sharpening)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (self.yPoints, self.xPoints), None)

            if (ret is True):

                self.leftImagesAll.append(img)
                self.leftImagesAllOrig.append(img_Orig)
                self.corners_l.append(corners)
                self.objpointsLeft.append(self.objp)

                rt = cv2.cornerSubPix(gray, corners, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners)

                # Draw and display the corners

                cv2.drawChessboardCorners(img, (self.yPoints, self.xPoints),corners, ret)
                # cv2.imshow(images[i], img)
                # cv2.waitKey(1500)
                # saveName =
                saveName =  cal_path +"/results/leftR/"+ os.path.basename(fname)
                print(saveName)
                cv2.imwrite(saveName,img)
            else:
                # Delete Other Images
                os.remove(images[i])



        img_shape = gray.shape[::-1]
        self.image_size_l = img_shape

        # time.sleep(10000)
        print("Left Camera Calibrating")

        rt, self.mLeft, self.dLeft, self.rLeft, self.tLeft = cv2.calibrateCamera(
            self.objpointsLeft, self.imgpoints_l, img_shape, None, None)

        print("----------------")
        print("Camera Matrix Left:")
        print(self.mLeft)

        print("----------------")
        print("Distortion Coefficients Left:")
        print(self.dLeft)

        camData = {
    		"cameraMatrixLeft" :self.mLeft,
            "distortionCoefficientsLeft" :self.dLeft,
    	     }

        saveName = "leftCamCalibration_" + str(datetime.datetime.now())
        with open(saveName, 'wb') as f:
            pickle.dump(camData, f)


        return True;

    def readRight(self, cal_path):

        print("Reading Right Images")

        images= glob.glob(cal_path + 'right/*.jpg')
        images.sort()


        for i, fname in enumerate(images):

            img = resizeImage(self,cv2.imread(images[i]),100)
            img_Orig = resizeImage(self,cv2.imread(images[i]),100)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (self.yPoints, self.xPoints), None)

            if (ret is True):

                self.rightImagesAll.append(img)
                self.rightImagesAllOrig.append(img_Orig)
                self.corners_r.append(corners)
                self.objpointsRight.append(self.objp)

                rt = cv2.cornerSubPix(gray, corners, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners)

                # Draw and display the corners

                cv2.drawChessboardCorners(img, (self.yPoints, self.xPoints),corners, ret)
                # cv2.imshow(images[i], img)
                # cv2.waitKey(1500)

                saveName =  cal_path +"/results/rightR/"+ os.path.basename(fname)
                print(saveName)
                cv2.imwrite(saveName,img)
            else:
                #Delete Other Images
                os.remove(images[i])

        img_shape = gray.shape[::-1]
        self.image_size_r = img_shape

        # time.sleep(10000)
        print("Right Camera Calibrating")

        rt, self.mRight, self.dRight, self.rRight, self.tRight = cv2.calibrateCamera(
            self.objpointsRight, self.imgpoints_r, img_shape, None, None)

        print("----------------")
        print("Camera Matrix Right:")
        print(self.mRight)
        print("----------------")
        print("Distortion Coefficients Right:")
        print(self.dRight)

        camData = {
		"cameraMatrixRight" :self.mRight,
        "distortionCoefficientsRight" :self.dRight,
	     }

        saveName = "rightCamCalibration_" + str(datetime.datetime.now())
        with open(saveName, 'wb') as f:
            pickle.dump(camData, f)


        return True;

    def readThermal(self, cal_path):

        print("Reading Thermal Images")

        images= glob.glob(cal_path + 'thermal/*.jpg')
        images.sort()


        for i, fname in enumerate(images):

            img = resizeImage(self,cv2.imread(images[i]),100)
            img_Orig = resizeImage(self,cv2.imread(images[i]),100)

            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # gray    = cv2.equalizeHist(grayPre)

            # applying the sharpening kernel to the input image & displaying it.
            # gray = cv2.filter2D(gray, -1, kernel_sharpening)


                    # Create our shapening kernel,

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (self.yPoints, self.xPoints), None)

            if (ret is True):

                self.thermalImagesAll.append(img)
                self.thermalImagesAllOrig.append(img_Orig)
                self.corners_t.append(corners)
                self.objpointsThermal.append(self.objp)

                rt = cv2.cornerSubPix(gray, corners, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_t.append(corners)

                # Draw and display the corners

                cv2.drawChessboardCorners(img, (self.yPoints, self.xPoints),corners, ret)
                # cv2.imshow(images[i], img)
                # cv2.waitKey(1500)
                saveName =  cal_path +"/results/thermalR/"+ os.path.basename(fname)
                print(saveName)
                cv2.imwrite(saveName,img)
            else:
                #Delete Other Images
                os.remove(images[i])

        img_shape = gray.shape[::-1]
        self.image_size_t = img_shape

        # time.sleep(10000)
        print("Thermal Camera Calibrating")

        rt, self.mThermal, self.dThermal, self.rThermal, self.tThermal = cv2.calibrateCamera(
            self.objpointsThermal, self.imgpoints_t, img_shape, None, None)

        print("----------------")
        print("Camera Matrix Thermal:")
        print(self.mThermal)

        print("----------------")
        print("Distortion Coefficients Thermal:")
        print(self.dThermal)

        camData = {
		"cameraMatrixThermal" :self.mThermal,
        "distortionCoefficientsThermal" :self.dThermal,
	     }

        saveName = "thermalCamCalibration_" + str(datetime.datetime.now())
        with open(saveName, 'wb') as f:
            pickle.dump(camData, f)

        return True;

if __name__ == '__main__':
    parser = readArgs()
    print(parser['filePath'])
    cal_data = minstThreeFoldCamParametors(parser['filePath'],\
                                            parser['xSquares']-1,\
                                            parser['ySquares']-1\
                                            )
