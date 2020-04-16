"""
Created by Sanjay at 4/10/2020

Feature: Enter feature name here
Enter feature description here
"""
import cv2
from source.utils.common_functions import report_results
import matplotlib.pyplot as plt
import numpy as np

from source.face.FisherFace import read_faces
from source.utils.utils import *

plt.style.use('seaborn-white')


if __name__ == '__main__':
    train_faces, train_y = read_faces(FACE_TRAIN_DIR, dont_squeeze=True)  # read faces, labels (train)
    test_faces, test_y = read_faces(FACE_TEST_DIR, dont_squeeze=True)  # read faces, labels (test)

    # print(train_faces.shape)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(train_faces, np.array(train_y))
    pred_y = []
    for test_img in test_faces:
        pred_y.append(model.predict(test_img)[0])

    score, confusion_matrix, report = report_results(test_y, np.array(pred_y), cmat_flag=True)
