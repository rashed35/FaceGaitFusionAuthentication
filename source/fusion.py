"""
Created by Sanjay at 4/14/2020

Feature: Enter feature name here
Enter feature description here
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from source.face.FisherFace import read_faces, myPCA
from source.tasks import question5, task8
from source.utils.common_functions import report_results
from source.utils.readData import read_data
from source.utils.signal_preprocess import get_train_test_features
from source.utils.utils import *

# plt.style.use('seaborn-white')
ids = ['04010', '04011', '04012', '04013', '04014', '04015', '04016', '04017', '04018', '04019']
action = 'walk'
sensor = 'Left_foot'


def learn_PCA(faces, _D=30):
    """
    Returns the PCA features, works as helper to myPCA method
    :param _D: number of eigenfaces to select
    :param faces: faces matrix (dimension: total_pixels x num_of_observations)
    :return: PCA features matrix (dimension: D x num_of_observations)
    """
    W, LL, m = myPCA(faces)  # PCs, eigenvalues, mean face
    W_e = W[:, :_D]  # select only 'D' eigenfaces
    m = np.expand_dims(m, 1)  # (xxx,) -> (xxx, 1)
    return W_e, m


def get_face_classifier():
    train_faces, train_y = read_faces(FACE_TRAIN_DIR)  # read faces, labels (train)
    test_faces, test_y = read_faces(FACE_TEST_DIR)  # read faces, labels (test)

    W_e, m = learn_PCA(train_faces)
    train_x = np.dot(W_e.T, (train_faces - m))  # calculating PCA features (train)
    test_x = np.dot(W_e.T, (test_faces - m))  # calculating PCA features (test)

    f_model = RandomForestClassifier(n_estimators=50)
    f_model.fit(train_x.T, train_y)
    # f_model = KNeighborsClassifier(n_neighbors=9)
    # f_model.fit(train_x.T, train_y)

    return f_model, test_x, test_y


def get_gait_classifier():
    gait_dataset = read_data(GAIT_DATA_HOME, ids, action, sensor)
    train_x, train_y, test_x, test_y = get_train_test_features(gait_dataset)
    g_model = RandomForestClassifier(n_estimators=150)
    g_model.fit(train_x, train_y)  # train
    # g_model = DecisionTreeClassifier()
    # g_model.fit(train_x, train_y)
    return g_model, test_x, test_y


if __name__ == '__main__':
    face_model, face_test_x, face_test_y = get_face_classifier()
    gait_model, gait_test_x, gait_test_y = get_gait_classifier()

    pred_face = face_model.predict_proba(face_test_x.T)
    pred_gait = gait_model.predict_proba(gait_test_x)
    pred_gait_avg = np.zeros((0, 10))
    for i in range(10):
        t2 = pred_gait[i * 8:(i + 1) * 8, :]  # pick 8 rows
        pred_gait_avg = np.vstack((pred_gait_avg, t2.mean(axis=0)))

    alpha_list = [x / 10.0 for x in range(7, 8)]
    acc_list = []
    for alpha in alpha_list:
        print('alpha=%.2f' % alpha)
        acc_count = 0
        pred_y = []
        for i in range(face_test_x.shape[1]):  # loop through all test face images
            true_y = face_test_y[i]
            pf = pred_face[i, :]  # face prediction vector of shape (10,)
            gf = pred_gait_avg[true_y, :]  # gait prediction vector of shape (10,)

            fused_pred_vector = alpha * pf + (1-alpha) * gf
            fused_pred = np.argmax(fused_pred_vector)
            pred_y.append(fused_pred)
            acc_count += 1 if fused_pred == true_y else 0
        acc, cm, rpt = report_results(face_test_y, pred_y, cmat_flag=True)
        task8(cm)

        acc = acc_count / face_test_x.shape[1]
        acc_list.append(acc)
        print('\tAccuracy: %.2f' % acc)
    # question5(alpha_list, acc_list)
