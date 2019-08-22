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
import os
import threading
import time



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
#


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
    capture = cv2.VideoCapture(indexIn)
    # capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
    # capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    time.sleep(0.1)
    ret0, out_image = capture.read()
    # print(out_image.shape)
    # time.sleep(1)
    # out_image = cv2.cvtColor(out_image , cv2.COLOR_HSV2BGR)

    # cv2.imshow('frame',out_image)
    # cv2.waitKey(0)


    capture.release()
    return ret0,out_image;


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


def saveWebCamImage(capture,imageName):
    ret,image  = capture.read()
    print(image)
    print(imageName)
    cv2.imwrite(imageName,image)



def main():
  ctx = POINTER(uvc_context)()
  dev = POINTER(uvc_device)()
  devh = POINTER(uvc_device_handle)()
  ctrl = uvc_stream_ctrl()

  res = libuvc.uvc_init(byref(ctx), 0)
  if res < 0:
    print("uvc_init error")
    exit(1)

  try:
    res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
    if res < 0:
      print("uvc_find_device error")
      exit(1)

    try:
      res = libuvc.uvc_open(dev, byref(devh))
      if res < 0:
        print("uvc_open error")
        exit(1)

      print("device opened!")

      print_device_info(devh)
      print_device_formats(devh)

      frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
      if len(frame_formats) == 0:
        print("device does not support Y16")
        exit(1)

      libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
        frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval)
      )

      res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
      if res <0:
        print("uvc_start_streaming failed: {0}".format(res))
        exit(1)


      myCmd = os.popen('v4l2-ctl --list-devices').read()

      leftCamIndex  = getLeftWebCamIndex(myCmd)[1]
      rightCamIndex  = getRightWebCamIndex(myCmd)[1]

      capLeft       = cv2.VideoCapture(leftCamIndex)
      # capLeft.set(cv2.CAP_PROP_FRAME_WIDTH,2048)
      # capLeft.set(cv2.CAP_PROP_FRAME_HEIGHT,1536)
      capRight      = cv2.VideoCapture(rightCamIndex)
      # capRight.set(cv2.CAP_PROP_FRAME_WIDTH,2048)
      # capRight.set(cv2.CAP_PROP_FRAME_HEIGHT,1536)

      # while(True):
      #       # Capture frame-by-frame
      #       ret, frame = cap.read()
      #
      #       # Our operations on the frame come here
      #       # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      #
      #       # Display the resulting frame
      #       cv2.imshow('frame',frame)
      #       if cv2.waitKey(1) & 0xFF == ord('q'):
      #           break
      #
      #   # When everything done, release the capture
      # cap.release()
      cv2.destroyAllWindows()





      try:
        startTime = time.time()
# your code

        while True:
          # # Thermal Camera Save
          retLeft, leftImage = capLeft.read()
          retRight,rightImage = capRight.read()
          #

          # cv2.imshow('left',leftImage)
          # cv2.imshow('right',rightImage)

          elapsedTime = time.time() - startTime
          # print(elapsedTime)
          if elapsedTime > 4:
              readTime = time.time()
              dateTime = datetime.datetime.now()
              leftImageName    = "threeWayImageDataSets/utdSet0/"+ getImagePathTail(dateTime,'left')
              rightImageName   = "threeWayImageDataSets/utdSet0/"+ getImagePathTail(dateTime,'right')
              thermalImageName = "threeWayImageDataSets/utdSet0/"+ getImagePathTail(dateTime,'thermal')

              thermalData = q.get(True, 500)
              retLeft, leftImage = capLeft.read()
              retRight, rightImage = capRight.read()
              print("Read Time Thermal= "+ str(time.time() - readTime))

              startTime = time.time()
              dateTime = datetime.datetime.now()
              if (thermalData is None) or (not(retLeft and retRight)) :
                break

              thermalData   = cv2.resize(thermalData[:,:], (640, 480))
              thermalImage  = raw_to_8bit(thermalData)

              cv2.imwrite(leftImageName,leftImage)
              cv2.imwrite(rightImageName,rightImage)
              cv2.imwrite(thermalImageName,thermalImage)


        capLeft.release()
        capRight.release()

        cv2.destroyAllWindows()
      finally:
        libuvc.uvc_stop_streaming(devh)

      print("done")
    finally:
      libuvc.uvc_unref_device(dev)
  finally:
    libuvc.uvc_exit(ctx)

if __name__ == '__main__':
  main()
