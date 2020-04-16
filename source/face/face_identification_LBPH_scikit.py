"""
Created by Sanjay at 4/16/2020

Feature: Enter feature name here
Enter feature description here
"""
import random
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from source.utils.common_functions import report_results
import matplotlib.pyplot as plt
import numpy as np
from source.face.FisherFace import read_faces
from source.utils.utils import *

plt.style.use('seaborn-white')
K_LIST = [x for x in range(3, 8, 2)]  # for k-nearest neighbors
N_ESTIMATORS = [25, 50, 100, 150, 200]  # for random forest
radius = 19  # settings for LBP
n_points = 8 * radius  # settings for LBP


def extract_lbph(img):
    lbp = local_binary_pattern(img, n_points, radius, 'uniform')
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, n_points + 3),
                             range=(0, n_points + 2))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


if __name__ == '__main__':
    train_faces, train_labels = read_faces(FACE_TRAIN_DIR, dont_squeeze=True)  # read faces, labels (train)
    test_faces, test_labels = read_faces(FACE_TEST_DIR, dont_squeeze=True)  # read faces, labels (test)

    train_x, train_y = [], []
    for img, lbl in zip(train_faces, train_labels):
        train_x.append(extract_lbph(img))
        train_y.append(lbl)
    print(len(train_x), len(train_y))

    test_x, test_y = [], []
    for img, lbl in zip(test_faces, test_labels):
        test_x.append(extract_lbph(img))
        test_y.append(lbl)
    print(len(test_x), len(test_y))

    # print('# ======================= K-Nearest Neighbors ========================= #')
    # acc_list = []
    # for k in K_LIST:
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     knn.fit(train_x, train_y)
    #     knn_pred_y = knn.predict(test_x)
    #     print('k-neighbors: %d' % k)
    #     score, confusion_matrix, report = report_results(test_y, knn_pred_y)
    #     acc_list.append(score)
    # print()

    # print('# ======================= Random Forest ========================= #')
    # acc_list_rf = []
    # for n in N_ESTIMATORS:
    #     rand_forest = RandomForestClassifier(n_estimators=n)
    #     rand_forest.fit(train_x, train_y)
    #     pred_y = rand_forest.predict(test_x)
    #     score, confusion_matrix, report = report_results(test_y, pred_y)
    #     acc_list_rf.append(score)
    #     print('Tree count: %d \t Accuracy: %.3f' % (n, score))
    # print()

    print('# ======================= SVC ========================= #')
    # c = list(zip(train_x, train_y))
    # random.shuffle(c)
    # train_x, train_y = zip(*c)
    svc = SVC(kernel='linear')
    svc.fit(train_x, train_y)
    pred_y = svc.predict(test_x)
    score, confusion_matrix, report = report_results(test_y, pred_y)
