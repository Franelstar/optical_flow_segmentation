import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--movie", required=True, help="Movie path")
parser.add_argument("-K", "--K", required=False, help="K(Default 3)", default=3, type=int)
parser.add_argument("-k", "--kernel", required=False, help="Valeur du noyau (Default 7)", default=7, type=int)
parser.add_argument("-s", "--save", required=False, help="Save result (Default False)", default=False, type=bool)

args = vars(parser.parse_args())

video = args["movie"]
save = args["save"]
cap = cv.VideoCapture(cv.samples.findFile(video))

ret, frame1 = cap.read()
shape = (frame1.shape[1], frame1.shape[0])
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)

hsv[...,1] = 255

K = args["K"]
kernel = np.ones((args["kernel"], args["kernel"]), np.uint8)  # Noyau

fourcc = cv.VideoWriter_fourcc(*'XVID')
name_result = 'plus_vite_seuillage_{}_{}'.format(K, args["kernel"])
if save:
    out = cv.VideoWriter(name_result + '.avi', fourcc, 9.0, (shape[0] * 2, shape[1] * 2))

while (1):
    ret, frame2 = cap.read()
    if not ret:
        break
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 10, 5, 7, 1.5, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    mag_img = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    ang_img = ang * 180 / np.pi

    hsv[..., 0] = 0
    hsv[..., 2] = 0

    # Gestion de la norme
    attempts = 20
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    vectorized = np.float32(mag_img.reshape(-1, 1))
    ret, label, center = cv.kmeans(vectorized, K, None, criteria, attempts, cv.KMEANS_PP_CENTERS)
    center = np.uint8(center)

    # On met le centre max à 255 et le reste à 0
    maximum = center.max()
    for i, v in enumerate(center):
        if v == [maximum]:
            center[i] = 255
        else:
            center[i] = 0
    res = center[label.flatten()]
    hsv[..., 2] = res.reshape((mag_img.shape))

    # gestion des angles
    hsv[..., 0] = np.array(
        [0 if (0 <= value < 90) else (60 if (90 <= value < 180) else (120 if (180 <= value < 270) else 30)) for value in
         ang_img.reshape(shape[0] * shape[1])]).reshape(shape[1], shape[0])

    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    opening = cv.morphologyEx(hsv[..., 2], cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

    # Segmentation de l'image originale
    frame2_seg = cv.cvtColor(frame2, cv.COLOR_BGR2HSV)
    frame2_seg[..., 1] = np.array(
        [255 if value == 255 else frame2_seg[..., 1].reshape(shape[0] * shape[1])[index] for index, value in
         enumerate(closing.reshape(shape[0] * shape[1]))]).reshape(shape[1], shape[0])
    frame2_seg[..., 0] = np.array([0 if (0 <= value < 90 and closing.reshape(shape[0] * shape[1])[indx] == 255) else (
        60 if (90 <= value < 180 and closing.reshape(shape[0] * shape[1])[indx] == 255) else (
            120 if (180 <= value < 270 and closing.reshape(shape[0] * shape[1])[indx] == 255) else (
                30 if closing.reshape(shape[0] * shape[1])[indx] == 255 else
                frame2_seg[..., 0].reshape(shape[0] * shape[1])[indx]))) for indx, value in
                                   enumerate(ang_img.reshape(shape[0] * shape[1]))]).reshape(shape[1], shape[0])
    frame2_seg[..., 2] = np.array(
        [255 if value == 255 else frame2_seg[..., 2].reshape(shape[0] * shape[1])[index] for index, value in
         enumerate(closing.reshape(shape[0] * shape[1]))]).reshape(shape[1], shape[0])
    frame2_seg = cv.cvtColor(frame2_seg, cv.COLOR_HSV2BGR)

    # Masque
    masque = cv.cvtColor(closing, cv.COLOR_GRAY2BGR)

    height, width, _ = np.array(frame2).shape

    final_1 = np.concatenate((frame2, bgr), axis=1)
    final_2 = np.concatenate((frame2_seg, masque), axis=1)
    final = np.concatenate((final_1, final_2), axis=0)

    cv.imshow('frame2', final)
    if save:
        out.write(final)

    # Hit 'q' on the keyboard to quit!
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    prvs = next

# Release handle to the webcam
cap.release()
if save:
    out.release()
cv.destroyAllWindows()