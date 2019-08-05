

import cv2


def takeWebcamImage():
    image = []
    found = False
    for x in range(6):
        try:
            print("--------------")
            print(x)
            cap0 = cv2.VideoCapture(x)
            print("webCamFound")
            ret0, frame0 = cap0.read()
            if(frame0.size>60000):
                print(frame0.size)

                image = frame0
                found = True
                cv2.imshow('frame', frame0)
                cv2.waitKey(1000)
                break
        except:
            print("No Webcam found Thus far")

    return found,image;

takeWebcamImage()
