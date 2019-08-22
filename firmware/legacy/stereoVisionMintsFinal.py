import numpy as np
import cv2
import glob
import argparse


def resizeImage(imgOrig,scalePercent):
    width  = int(imgOrig.shape[1] * scalePercent / 100)
    height = int(imgOrig.shape[0] * scalePercent / 100)
    dim = (width, height)
    Orig = cv2.resize(imgOrig, dim, interpolation = cv2.INTER_AREA)
    return(Orig);


class StereoCalibration(object):
    def __init__(self, filepath):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((6*9, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        # Additions by Lakitha
        # Have a Generic Size
        self.img_size    = []
        self.rotational_matrix = []
        self.translational_matrix = []
        self.img_size    = []

        self.leftImagesAll  = []
        self.rightImagesAll = []
        self.corners_l = []
        self.corners_r = []

        self.cal_path = filepath
        self.read_images(self.cal_path)



    def read_images(self, cal_path):
        print("Reading Images")
        images_right = glob.glob(cal_path + 'thermal/*.jpg')
        images_left = glob.glob(cal_path + 'webCam/*.jpg')
        images_left.sort()
        images_right.sort()


        for i, fname in enumerate(images_right):

            img_l = resizeImage(cv2.imread(images_left[i]),100)
            img_r = resizeImage(cv2.imread(images_right[i]),100)


            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (6, 9), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (6, 9), None)

            # imgCor = cv2.drawChessboardCorners(gray_l, (6,9), corners_l,ret_l)
            #
            # cv2.imshow('left',imgCor)
            # cv2.waitKey(5000)
            #
            # imgCor = cv2.drawChessboardCorners(gray_r, (6,9), corners_r,ret_r)
            # print(imgCor)
            # cv2.imshow('left',imgCor)
            # cv2.waitKey(5000)

            # If found, add object points, image points (after refining them)

            self.objpoints.append(self.objp)

            if (ret_l is True) and (ret_r is True):

                self.leftImagesAll.append(img_l)
                self.rightImagesAll.append(img_r)

                self.corners_l.append(corners_l)
                self.corners_r.append(corners_r)
                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners

                ret_l = cv2.drawChessboardCorners(img_l, (6, 9),
                                                  corners_l, ret_l)
                cv2.imshow(images_left[i], img_l)
                # cv2.waitKey(10000)

            # if ret_r is True:
                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, (6, 9),
                                                  corners_r, ret_r)
                cv2.imshow(images_right[i], img_r)
                # cv2.waitKey(10000)

            img_shape = gray_l.shape[::-1]
            self.image_size = img_shape
        cv2.waitKey(0)
        print(self.image_size)
        print(ret_l)
        print(ret_r)
        print("Here")

        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_shape, None, None)

        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_shape, None, None)

        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5
        print("M1:")
        print(self.M1)
        print("----------------")
        print("M2:")
        print(self.M2)
        print("----------------")
        print(dims)
        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)

        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)

        print("M1Result:")
        print(M1)
        print("----------------")
        print("M2Result:")
        print(M2)
        print("----------------")
        print("-------------------")
        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)
        print('')

        self.rotational_matrix    = R
        self.translational_matrix = T

        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                            ('dist2', d2), ('rvecs1', self.r1),
                            ('rvecs2', self.r2), ('R', R), ('T', T),
                            ('E', E), ('F', F)])




        cv2.destroyAllWindows()
        return camera_model;


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    cal_data = StereoCalibration(args.filepath)
    print("--------------------")

    print(cal_data.corners_l[0][0])
    print(cal_data.corners_r[0][0])



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
