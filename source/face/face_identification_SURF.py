"""
Created by Sanjay at 4/10/2020

Feature: Enter feature name here
Enter feature description here
"""
import random

import cv2
from sklearn.svm import SVC

from source.common_functions import report_results_face
import numpy as np
from source.face.FisherFace import read_faces
from source.utils.utils import *


def extract_surf(img):
    """
    Extract SURF keypoints and descriptors from an image
    """
    # surf = cv2.SURF(hessianThreshold=hessian, nOctaves=octave, nOctaveLayers=octaveLayers, extended=ext)
    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(img, None)

    return kp, des


if __name__ == '__main__':
    train_faces, train_labels = read_faces(FACE_TRAIN_DIR, dont_squeeze=True)  # read faces, labels (train)
    test_faces, test_labels = read_faces(FACE_TEST_DIR, dont_squeeze=True)  # read faces, labels (test)

    train_x, train_y = [], []
    for img, lbl in zip(train_faces, train_labels):
        key_points, descriptors = extract_surf(img)
        for d in descriptors:
            train_x.append(d)
            train_y.append(lbl)

    c = list(zip(train_x, train_y))
    random.shuffle(c)
    train_x, train_y = zip(*c)

    svc = SVC(kernel='rbf', C=10000, gamma=0.1)
    svc.fit(train_x, train_y)

    test_y, pred_y = [], []
    for img, lbl in zip(test_faces, test_labels):
        key_points, descriptors = extract_surf(img)
        pred_list = svc.predict(descriptors)
        counts = np.bincount(pred_list)  # bin counting for upcoming argmax
        pred = np.argmax(counts)
        pred_y.append(pred)
        test_y.append(lbl)

    score, confusion_matrix, report = report_results_face(test_y, np.array(pred_y), cmat_flag=True)
