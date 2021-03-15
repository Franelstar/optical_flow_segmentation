import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--movie", required=True, help="Movie path")

args = vars(parser.parse_args())

video = args["movie"]

cap = cv.VideoCapture(cv.samples.findFile(video))

ret, frame1 = cap.read()

prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

hsv = np.zeros_like(frame1)

hsv[..., 1] = 255

while (1):
    ret, frame2 = cap.read()
    if ret == False:
        break
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    height, width, _ = np.array(frame2).shape
    pourcentage_h = 0.7
    pourcentage_w = 0.4

    final = vis = np.concatenate((cv.resize(frame2, (int(height * pourcentage_h), int(width * pourcentage_w))),
                                  cv.resize(bgr, (int(height * pourcentage_h), int(width * pourcentage_w)))), axis=1)

    cv.imshow('frame2', final)

    # Hit 'q' on the keyboard to quit!
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    if cv.waitKey(1) & 0xFF == ord('a'):
        cv.imwrite('opticalfc_2.png', final)

    prvs = next

# Release handle to the webcam
cap.release()
cv.destroyAllWindows()