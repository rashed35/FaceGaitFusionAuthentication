"""
Created by Sanjay at 4/8/2020

Feature: Enter feature name here
Enter feature description here
"""

import cv2
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from source.tasks import question3
from source.utils.common_functions import report_results
from source.face.FisherFace import read_faces, myPCA
from source.utils.utils import *
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

plt.style.use('seaborn-white')

# ======================= SETTINGS ========================= #
D = 30  # Number of eigenfaces, for PCA
D_LIST = [10, 20, 30, 40, 50, 70, 100]
K_LIST = [x for x in range(7, 8, 2)]  # for k-nearest neighbors
N_ESTIMATORS = [25, 50, 100, 200]  # for random forest


def back_projection(f, W_e, m, img_name, do_save=False):
    """
    Back projects eigenfaces to 2D image
    :param f: face in PCA space (D dimension)
    :param W_e: Eigenfaces (D columns)
    :param m: mean face
    :param img_name: label of the face
    :param do_save: flag to know if file should be saved
    :return: None
    """
    _x = np.dot(W_e, f) + m  # back projection of face
    _x = _x.reshape(IMG_SHAPE)  # reshape to 2D image
    if do_save:
        save_dir = os.path.join(PCA_BACK_PROJECT_DIR, 'D_' + str(D))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(PCA_BACK_PROJECT_DIR, 'D_' + str(D), str(img_name)), _x)


def learn_PCA(faces, _D=D):
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


if __name__ == '__main__':
    train_faces, train_y = read_faces(FACE_TRAIN_DIR)  # read faces, labels (train)
    test_faces, test_y = read_faces(FACE_TEST_DIR)  # read faces, labels (test)

    W_e, m = learn_PCA(train_faces, D)
    train_x = np.dot(W_e.T, (train_faces - m))  # calculating PCA features (train)
    test_x = np.dot(W_e.T, (test_faces - m))  # calculating PCA features (test)

    # print('# ======================= Question 3 ========================= #')
    # acc_list = []
    # cm_list = []
    # for d in D_LIST:
    #     W_e, m = learn_PCA(train_faces, d)
    #     train_x = np.dot(W_e.T, (train_faces - m))  # calculating PCA features (train)
    #     test_x = np.dot(W_e.T, (test_faces - m))  # calculating PCA features (test)
    #
    #     rand_forest = RandomForestClassifier(n_estimators=50)
    #     rand_forest.fit(train_x.T, train_y)
    #     pred_y = rand_forest.predict(test_x.T)
    #     print('D = ', d)
    #     acc, cm, rpt = report_results_face(test_y, pred_y, cmat_flag=True)
    #     acc_list.append(acc)
    #
    #     # df_cm = pd.DataFrame(cm, index=[i for i in range(10)], columns=[i for i in range(10)])
    #     # plt.figure(figsize=(10, 7))
    #     # plt.title('Confusion Matrix for Random Forest | n=50, D='+str(d)+' (for PCA)')
    #     # sn.heatmap(df_cm, annot=True)
    #     # plt.savefig(os.path.join(RESULTS_HOME, 'question4_confusion_matrix_D_'+str(d)+'.png'), dpi=300)
    #     # # plt.show()
    # question3(D_LIST, acc_list)

    print('# ======================= K-Nearest Neighbors ========================= #')
    acc_list = []
    cmat_k5 = None
    for k in K_LIST:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_x.T, train_y)
        knn_pred_y = knn.predict(test_x.T)
        print('k-neighbors: %d' % k)

        acc, cmat, rpt = report_results(test_y, knn_pred_y) if k != 5 \
            else report_results(test_y, knn_pred_y, cmat_flag=True)
        cmat_k5 = cmat if k == 5 else cmat_k5
        acc_list.append(acc)
    print()

    print('# ======================= Random Forest ========================= #')
    acc_list_rf = []
    cmat_n100 = None
    for n in N_ESTIMATORS:
        rand_forest = RandomForestClassifier(n_estimators=n)
        rand_forest.fit(train_x.T, train_y)
        rf_pred_y = rand_forest.predict(test_x.T)
        score = metrics.accuracy_score(test_y, rf_pred_y)
        acc, cmat, rpt = report_results(test_y, rf_pred_y) if n != 100 \
            else report_results(test_y, rf_pred_y, cmat_flag=True)
        cmat_n100 = cmat if n == 100 else cmat_n100
        acc_list_rf.append(acc)
        print('Tree count: %d \t Accuracy: %.3f' % (n, score))
    print()

    print('# ======================= SVC ========================= #')
    svc = SVC(kernel='linear', C=1)
    svc.fit(train_x.T, train_y)
    pred_y = svc.predict(test_x.T)
    score, confusion_matrix, report = report_results(test_y, np.array(pred_y), cmat_flag=True)

    # ======================= TASKS ========================= #
    # task2(W_e, m)
    # task3(K_LIST, acc_list, cmat_k5)
    # acc_list_rf, cmat_n100 = np.load(os.path.join(RESULTS_HOME, 'saved', 'acc_list_rf.npy')),
    # np.load(os.path.join(RESULTS_HOME, 'saved', 'cmat_n100.npy'))  # pre-saved results for task4
    # task4(N_ESTIMATORS, acc_list_rf, cmat_n100)
