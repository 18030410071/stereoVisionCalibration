#
# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# import datetime
# from uvctypes import *
# import time
import cv2
# import numpy as np
# try:
#   from queue import Queue
# except ImportError:
#   from Queue import Queue
# import platform
#
import pygame
import os

def getLeftWebCamIndex(myCmd):
    lines = myCmd.splitlines()
    for num, name in enumerate(lines, start=1):
        if "7-4.4" in name:
            camString =  int(lines[num].strip().replace("/dev/video",""))
            return  True ,camString;

    return False , "xxx";


def getRightWebCamIndex(myCmd):
    lines = myCmd.splitlines()
    for num, name in enumerate(lines, start=1):
        if "7-4.2" in name:
            camString =  int(lines[num].strip().replace("/dev/video",""))
            return True, camString;

    return False, "xxx";


def takeWebCamImage(indexIn):
    cap0 = cv2.VideoCapture(indexIn)
    ret0, frame0 = cap0.read()
    return ret0,frame0;

# myCmd = os.popen('v4l2-ctl --list-devices').read()
# print(myCmd)
# leftImage      = takeWebCamImage(getLeftWebCamIndex(myCmd)[1])
# rightImage     = takeWebCamImage(getRightWebCamIndex(myCmd)[1])
#
# # rightImage    = takeWebCamImage(1)
# # # leftImageRGB  = cv2.cvtColor(leftImage[0], cv2.COLOR_BGR2RGB)
# # # rightImageRGB = cv2.cvtColor(rightImage[0], cv2.COLOR_BGR2RGB)
#
# cv2.imwrite("left1.jpg",leftImage[1])#
# cv2.imwrite("right1.jpg",rightImage[1])#
#
#
# pygame.camera.init()
# pygame.camera.list_cameras()
# cam = pygame.camera.Camera("/dev/video6", (640, 480))
# cam.start()
# time.sleep(0.1)  # You might need something higher in the beginning
# img = cam.get_image()
# pygame.image.save(img, "pygame.jpg")
# cam.stop()

import numpy as np
import cv2

cap = cv2.VideoCapture(6)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# import pygame
# import pygame.camera
#
# pygame.camera.init()
# # pygame.camera.list_camera() #Camera detected or not
# cam = pygame.camera.Camera("/dev/video0",(640,480))
# cam.start()
# img = cam.get_image()
# pygame.image.save(img,"filename.jpg")




# def getWebCamIndex():
#     image = []
#     found = False
#     for x in range(6):
#         try:
#
#             cap0 = cv2.VideoCapture(x)
#             ret0, frame0 = cap0.read()
#             if(frame0.size>60000):
#                 index = x
#                 found = True
#                 break
#         except:
#             print("No Webcam found Thus far")
#
#     return found,index;
# import device
# import cv2
#
# def select_camera(last_index):
#     number = 0
#     hint = "Select a camera (0 to " + str(last_index) + "): "
#     # try:
#     number = int(input(hint))
#         # select = int(select)
#     # except Exception ,e:
#     #     print("It's not a number!")
#     #     return select_camera(last_index)
#
#     if number > last_index:
#         print("Invalid number! Retry!")
#         return select_camera(last_index)
#
#     return number
#
#
# def open_camera(index):
#     cap = cv2.VideoCapture(index)
#     return cap
#
# def main():
#     # print OpenCV version
#     print("OpenCV version: " + cv2.__version__)
#
#     # Get camera list
#     device_list = device.getDeviceList()
#     index = 0
#
#     for name in device_list:
#         print(str(index) + ': ' + name)
#         index += 1
#     #
    # last_index = index - 1
    #
    # if last_index < 0:
    #     print("No device is connected")
    #     return
    #
    # # Select a camera
    # camera_number = select_camera(last_index)
    #
    # # Open camera
    # cap = open_camera(camera_number)
    #
    # if cap.isOpened():
    #     width = cap.get(3) # Frame Width
    #     height = cap.get(4) # Frame Height
    #     print('Default width: ' + str(width) + ', height: ' + str(height))
    #
    #     while True:
    #
    #         ret, frame = cap.read();
    #         cv2.imshow("frame", frame)
    #
    #         # key: 'ESC'
    #         key = cv2.waitKey(20)
    #         if key == 27:
    #             break
    #
    #     cap.release()
    #     cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     main()
