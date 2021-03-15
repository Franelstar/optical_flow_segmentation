import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--movie", required=True, help="Movie path")
parser.add_argument("-t1", "--threshold_1", required=False, help="Valeur de seuil (Default 70)", default=70, type=int)
parser.add_argument("-t2", "--threshold_2", required=False, help="Valeur de seuil (Default 150)", default=150, type=int)
parser.add_argument("-k", "--kernel", required=False, help="Valeur du noyau (Default 7)", default=7, type=int)
parser.add_argument("-s", "--save", required=False, help="Save result (Default False)", default=False, type=bool)

args = vars(parser.parse_args())

video = args["movie"]
save = args["save"]
cap = cv.VideoCapture(cv.samples.findFile(video))

ret, frame1 = cap.read()
shape = (frame1.shape[1], frame1.shape[0])
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
lab = np.zeros_like(frame1)

seuil_1 = args["threshold_1"]
seuil_2 = args["threshold_2"]
kernel = np.ones((args["kernel"], args["kernel"]), np.uint8)  # Noyau

fourcc = cv.VideoWriter_fourcc(*'XVID')
name_result = 'plus_vite_seuillage_{}_{}_{}'.format(seuil_1, seuil_2, args["kernel"])
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

    lab[..., 0] = 0
    lab[..., 1] = 128
    lab[..., 2] = 128

    ret, lab[..., 0] = cv.threshold(mag_img, seuil_2, 255, cv.THRESH_BINARY)
    ret, lab[..., 1] = cv.threshold(mag_img, seuil_2, 255, cv.THRESH_BINARY)
    lab[..., 1] = np.array(
        [128 if value == 0 else value for value in lab[..., 1].reshape(shape[0] * shape[1])]).reshape(shape[1],
                                                                                                      shape[0])
    ret, lab[..., 0] = cv.threshold(mag_img, seuil_1, 255, cv.THRESH_BINARY)
    ret, lab[..., 2] = cv.threshold(mag_img, seuil_1, 255, cv.THRESH_BINARY)
    lab[..., 2] = np.array(
        [128 if value == 0 else value for value in lab[..., 2].reshape(shape[0] * shape[1])]).reshape(shape[1],
                                                                                                      shape[0])

    bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)

    opening_1 = cv.morphologyEx(lab[..., 1], cv.MORPH_OPEN, kernel)  # Ouverture
    closing_1 = cv.morphologyEx(opening_1, cv.MORPH_CLOSE, kernel)  # Fermerture

    opening_2 = cv.morphologyEx(lab[..., 2], cv.MORPH_OPEN, kernel)  # Ouverture
    closing_2 = cv.morphologyEx(opening_2, cv.MORPH_CLOSE, kernel)  # Fermerture

    # Segmentation de l'image originale
    frame2_seg = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)
    frame2_seg[..., 0] = np.array(
        [255 if value == 255 else frame2_seg[..., 0].reshape(shape[0] * shape[1])[index] for index, value in
         enumerate(closing_1.reshape(shape[0] * shape[1]))]).reshape(shape[1], shape[0])
    frame2_seg[..., 2] = np.array([255 if value == 255 and frame2_seg[..., 0].reshape(shape[0] * shape[1])[
        index] != 255 else frame2_seg[..., 2].reshape(shape[0] * shape[1])[index] for index, value in
                                   enumerate(closing_2.reshape(shape[0] * shape[1]))]).reshape(shape[1], shape[0])
    frame2_seg = cv.cvtColor(frame2_seg, cv.COLOR_RGB2BGR)

    # Masque
    masque = closing_1 + closing_2
    masque = cv.cvtColor(masque, cv.COLOR_GRAY2BGR)

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