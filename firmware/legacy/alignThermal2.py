#!/usr/bin/env python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
from uvctypes import *
import time
import cv2
import numpy as np
try:
  from queue import Queue
except ImportError:
  from Queue import Queue
import platform
import pickle
BUF_SIZE = 2
q = Queue(BUF_SIZE)

def py_frame_callback(frame, userptr):

  array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
  data = np.frombuffer(
    array_pointer.contents, dtype=np.dtype(np.uint16)
  ).reshape(
    frame.contents.height, frame.contents.width
  ) # no copy

  # data = np.fromiter(
  #   frame.contents.data, dtype=np.dtype(np.uint8), count=frame.contents.data_bytes
  # ).reshape(
  #   frame.contents.height, frame.contents.width, 2
  # ) # copy

  if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
    return

  if not q.full():
    q.put(data)

PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)

def ktof(val):
  return (1.8 * ktoc(val) + 32.0)

def ktoc(val):
  return (val - 27315) / 100.0

def raw_to_8bit(data):
  cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
  np.right_shift(data, 8, data)
  return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)

def display_temperature(img, val_k, loc, color):
  val = ktoc(val_k)
  cv2.putText(img,"{0:.1f} degC".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
  x, y = loc
  cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
  cv2.line(img, (x, y - 2), (x, y + 2), color, 1)



def getWebCamIndex():
    image = []
    found = False
    for x in range(6):
        try:

            cap0 = cv2.VideoCapture(x)
            ret0, frame0 = cap0.read()
            if(frame0.size>60000):
                index = x
                found = True
                break
        except:
            print("No Webcam found Thus far")

    return found,index;


def takeWebCamImage(indexIn):
    cap0 = cv2.VideoCapture(indexIn)
    ret0, frame0 = cap0.read()
    print(ret0)

    return ret0,frame0;

def getImagePathTail(dateTime,labelIn):
    pathTail = labelIn+"/"+\
    str(dateTime.year).zfill(4) + \
    "_" +str(dateTime.month).zfill(2) + \
    "_" +str(dateTime.day).zfill(2)+ \
    "_" +str(dateTime.hour).zfill(2) + \
    "_" +str(dateTime.minute).zfill(2)+ \
    "_" +str(dateTime.second).zfill(2)+ \
    "_"+labelIn+".jpg"
    print(pathTail)
    return pathTail;




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


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

def alignImages(im1, im2):

  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)

  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)

  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))

  return im1Reg, h



def main():

    # thermalData = cv2.resize(thermalData[:,:], (640, 480))
    # thermalImage = raw_to_8bit(thermalData)
    # dateTime  =

    im1 =  cv2.imread("stereoImageDataSets/thermalAndWebCam/webCam/2019_08_05_10_02_44_webCam.jpg");
    im2 =  cv2.imread("stereoImageDataSets/thermalAndWebCam/thermal/2019_08_05_10_02_44_thermal.jpg");


    # webCamImageName  = "outPuts/"+ getImagePathTail(dateTime,'webCam')
    # thermalImageName = "outPuts/"+ getImagePathTail(dateTime,'thermal')
    # #
    # cv2.imwrite(webCamImageName,webCamImage)
    # cv2.imwrite(thermalImageName,thermalImage)

    with open('stereoDataAug4.pickle', 'rb') as f:
         stereoData = pickle.load(f)

    print(stereoData)
    print(stereoData["cameraMatrixLeft"])

          #
    undistWebCam  = cv2.undistort(im1, stereoData['cameraMatrixLeft'], stereoData['distortionCoefficientsLeft'], None,stereoData['cameraMatrixLeft'])
    # cv2.imshow("Left", undistWebCam)
    undistThermal = cv2.undistort(im2, stereoData['cameraMatrixRight'], stereoData['distortionCoefficientsRight'], None,stereoData['cameraMatrixRight'])

    cv2.imshow("Undist WebCamImage", undistWebCam)
    cv2.imshow("Thermal", undistThermal)
    cv2.waitKey(10000)

    # Use Image Restoration
    # Read the images to be aligned

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(undistWebCam ,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(undistThermal,cv2.COLOR_BGR2GRAY)

    # # Find size of image1
    # sz = im1.shape
    #
    # # Define the motion model
    # warp_mode = cv2.MOTION_TRANSLATION
    #
    # # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    # if warp_mode == cv2.MOTION_HOMOGRAPHY :
    #     warp_matrix = np.eye(3, 3, dtype=np.float32)
    # else :
    #     warp_matrix = np.eye(2, 3, dtype=np.float32)
    #
    # # Specify the number of iterations.
    # number_of_iterations = 5000;
    #
    # # Specify the threshold of the increment
    # # in the correlation coefficient between two iterations
    # termination_eps = 1e-10;
    #
    # # Define termination criteria
    # criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    #
    # # Run the ECC algorithm. The results are stored in warp_matrix.
    # (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
    #
    # if warp_mode == cv2.MOTION_HOMOGRAPHY :
    #     # Use warpPerspective for Homography
    #     im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    # else :
    #     # Use warpAffine for Translation, Euclidean and Affine
    #     im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    #
    # # Show final results
    # cv2.imshow("Image 1", im1)
    # cv2.imshow("Image 2", im2)
    # cv2.waitKey(10000)
    #
    # cv2.imshow("Aliged Image Webcam", undistWebCam)
    # cv2.imshow("Aligned Image 2", im2_aligned)
    # cv2.waitKey(10000)
    # # cv2.imshow("Aligned Image 2", im2_aligned)
    # cv2.waitKey(0)


    croppedWebCam  = four_point_transform(undistWebCam, stereoData['cropPointsLeft'])
    croppedThermal = four_point_transform(undistThermal, stereoData['cropPointsRight'])

    #       #
    # print("Webcam Image Size")
    # print(croppedWebCam.shape)
    #
    # print("Thermal Image Size")
    # print(croppedThermal.shape)

    # # Converting to Greyx
    cv2.imshow("Cropped WebCam", croppedWebCam)
    cv2.imshow("Cropped Thermal",croppedThermal)
    cv2.waitKey(20000)

    #       # print(resizedThermal.shape)
    resizedWebCam = cv2.resize(croppedWebCam,\
                                    (croppedThermal.shape[1],croppedThermal.shape[0]),\
                                     interpolation = cv2.INTER_AREA)
    resizedThermal  = cv2.cvtColor(croppedThermal, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Resized WebCam" , resizedWebCam )
    cv2.imshow("Resized Thermal", resizedThermal)

    cv2.waitKey(20000)


    #
    #       greyScaleWebCam      =  cv2.cvtColor(croppedWebCam, cv2.COLOR_BGR2GRAY)
    #       greyScaleThermal     =  cv2.cvtColor(croppedThermal, cv2.COLOR_BGR2GRAY)
    #
    #
    #         # Convert images to grayscale
    #       im1_gray = greyScaleWebCam
    #       im2_gray = greyScaleThermal
    #
    #         # Find size of image1
    #       sz =  resizedWebCam.shape
    #
    #         # Define the motion model
    #       warp_mode = cv2.MOTION_TRANSLATION
    #
    #         # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    #       if warp_mode == cv2.MOTION_HOMOGRAPHY :
    #           warp_matrix = np.eye(3, 3, dtype=np.float32)
    #       else :
    #           warp_matrix = np.eye(2, 3, dtype=np.float32)
    #
    #         # Specify the number of iterations.
    #       number_of_iterations = 5000;
    #
    #         # Specify the threshold of the increment
    #         # in the correlation coefficient between two iterations
    #       termination_eps = 1e-10;
    #
    #         # Define termination criteria
    #       criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    #
    #         # Run the ECC algorithm. The results are stored in warp_matrix.
    #       (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
    #       print()
    #
    #       if warp_mode == cv2.MOTION_HOMOGRAPHY :
    #             # Use warpPerspective for Homography
    #           im2_aligned = cv2.warpPerspective ( resizedThermal, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    #       else :
    #             # Use warpAffine for Translation, Euclidean and Affine
    #           im2_aligned = cv2.warpAffine( resizedThermal, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    #
    #         # Show final results
    #       cv2.imshow("Image 1", resizedWebCam )
    #       cv2.imshow("Image 2", resizedThermal)
    #       cv2.imshow("Aligned Image 2", im2_aligned)
    #       cv2.waitKey(0)
    #
    #

      #     cv2.imshow("GreyScale WebCam", greyScaleWebCam)
      #     cv2.imshow("GreyScale Thermal",greyScaleThermal)
      #
      #
      #     # AlignedWebCam, h = alignImages(resizedWebCam, croppedThermal)
      #     #
      #     #
      #     # cv2.imshow("Aligned WebCam", AlignedWebCam)
      #     # cv2.imshow("Aligned Thermal", resizedThermal)
      #
      #     cv2.waitKey(10000)
      # #
      #
      #
      #     # Create the haar cascade
      #     faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
      #
      #     facesWebCam = faceCascade.detectMultiScale(
      #                                   greyScaleWebCam,
      #                                   scaleFactor=1.1,
      #                                   minNeighbors=5,
      #                                   minSize=(30, 30),
      #                                   flags = cv2.CASCADE_SCALE_IMAGE
      #                               )
      #
      #     facesThermal= faceCascade.detectMultiScale(
      #                                   255-greyScaleThermal,
      #                                   scaleFactor=1.01,
      #                                   minNeighbors=2,
      #                                   minSize=(30, 30),
      #                                   flags = cv2.CASCADE_SCALE_IMAGE
      #                               )
      #
      #     # Draw a rectangle around the faces
      #
      #     print(facesWebCam)
      #     print(facesThermal)
      #
      #
      #     for (x, y, w, h) in facesWebCam:
      #         cv2.rectangle(resizedWebCam, (x, y), (x+w, y+h), (0, 255, 0), 2)
      #         break
      #     cv2.imshow("Faces WebCam", resizedWebCam)
      #
      #     for (x, y, w, h) in facesThermal:
      #         cv2.rectangle(resizedThermal, (x, y), (x+w, y+h), (0, 255, 0), 2)
      #         break
      #     cv2.imshow("Faces Thermal", resizedThermal)
      #
      #     cv2.waitKey(50000)
      #     cv2.destroyAllWindows()
      #               #
      #     time.sleep(2)
      #     # cv2.waitKey(0)


  #
  #       cv2.destroyAllWindows()
  #     finally:
  #       libuvc.uvc_stop_streaming(devh)
  #
  #     print("done")
  #   finally:
  #     libuvc.uvc_unref_device(dev)
  # finally:
  #   libuvc.uvc_exit(ctx)

if __name__ == '__main__':
  main()
