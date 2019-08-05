import numpy as np
import cv2
import glob
import argparse


def resizeImage(imgOrig,scalePercent):
    width  = int(imgOrig.shape[1] * scalePercent / 60)
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

        self.cal_path = filepath
        self.read_images(self.cal_path)



    def read_images(self, cal_path):
        print("Reading Images")
        images_right = glob.glob(cal_path + 'RIGHT/*.jpg')
        images_left = glob.glob(cal_path + 'LEFT/*.jpg')
        images_left.sort()
        images_right.sort()


        for i, fname in enumerate(images_right):

            img_l = resizeImage(cv2.imread(images_left[i]),40)
            img_r = resizeImage(cv2.imread(images_right[i]),40)

            self.leftImagesAll.append(img_l)
            self.rightImagesAll.append(img_r)

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

            if ret_l is True:
                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (6, 9),
                                                  corners_l, ret_l)
                cv2.imshow(images_left[i], img_l)
                cv2.waitKey(1000)

            if ret_r is True:
                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, (6, 9),
                                                  corners_r, ret_r)
                cv2.imshow(images_right[i], img_r)
                cv2.waitKey(1000)

            img_shape = gray_l.shape[::-1]
            self.image_size = img_shape
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    cal_data = StereoCalibration(args.filepath)
    # img = cal_data.leftImagesAll[4]
    # h, w = img.shape[:2]
    # newcameramtx, roi=cv2.getOptimalNewCameraMatrix(cal_data.M1,cal_data.d1,(w,h),1,(w,h))
    # dst = cv2.undistort(img, cal_data.M1, cal_data.d1, None, newcameramtx)
    # # crop the image
    # print(roi)
    # x,y,w,h = roi
    #
    # dst = dst[y:y+h, x:x+w]
    # print(dst)
    # cv2.imwrite('calibresult.png',dst)
