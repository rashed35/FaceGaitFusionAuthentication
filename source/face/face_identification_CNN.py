"""
Created by Sanjay at 4/15/2020

Feature: Enter feature name here
Enter feature description here
"""
import face_recognition as fr
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from source.face.FisherFace import read_faces
from source.tasks import task5
from source.utils.common_functions import report_results
from source.utils.utils import *

K_LIST = [x for x in range(3, 10, 2)]  # for k-nearest neighbors
N_ESTIMATORS = [25, 50, 100, 200]  # for random forest


def learn_CNN_features(img_dir):
    features = None
    labels = []  # Label will store list of identity label
    for filename in os.listdir(img_dir):
        if not filename[-3:] == 'bmp':
            continue
        img = fr.load_image_file(os.path.join(img_dir, filename))
        enc = None
        try:
            enc = fr.face_encodings(img)[0]
            name = filename.split('_')[0][-1]
            labels.append(int(name))
        except Exception:
            # enc = np.zeros((128,))
            print('zero ', filename)

        features = enc if features is None else np.vstack((features, enc))

    return features, np.array(labels)


def run_kNN():
    print('# ======================= K-Nearest Neighbors ========================= #')
    acc_list = []
    for k in K_LIST:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_x, train_y)
        knn_pred_y = knn.predict(test_x)
        print('k-neighbors: %d' % k)
        acc, cmat, rpt = report_results(test_y, knn_pred_y)
        acc_list.append(acc)
    print()


def run_RandomForest():
    print('# ======================= Random Forest ========================= #')
    acc_list_rf = []
    for n in [25]:
        rand_forest = RandomForestClassifier(n_estimators=n)
        rand_forest.fit(train_x, train_y)
        rf_pred_y = rand_forest.predict(test_x)
        print('Tree count: %d' % n)
        acc, cm, rpt = report_results(test_y, rf_pred_y, cmat_flag=True)
        acc_list_rf.append(acc)
        task5(cm, 'Random Forest (25 trees)')
    print()


def run_SVC():
    print('# ======================= SVC ========================= #')
    svc = SVC(kernel='linear', C=1)
    svc.fit(train_x, train_y)
    pred_y = svc.predict(test_x)
    score, confusion_matrix, report = report_results(test_y, np.array(pred_y), cmat_flag=True)
    print()


if __name__ == '__main__':
    """ 
    Different directories as cnn feature learner can't detect some faces
    Hence, those faces were removed to make new directories for CNN features
    """
    # train_x, train_y = learn_CNN_features(os.path.join(FACE_DATA_HOME, 'train.cnn'))
    # test_x, test_y = learn_CNN_features(os.path.join(FACE_DATA_HOME, 'test.cnn'))

    # loading pre-calculated features to save computation time
    train_x, train_y = np.load(r'.\cnn_features\train_enc.npy'), np.load(r'.\cnn_features\train_y.npy')
    test_x, test_y = np.load(r'.\cnn_features\test_enc.npy'), np.load(r'.\cnn_features\test_y.npy')

    print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

    # run_kNN()
    run_RandomForest()
    # run_SVC()
