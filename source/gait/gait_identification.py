"""
Created by Sanjay at 4/11/2020

Feature: Enter feature name here
Enter feature description here
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np

from source.utils.common_functions import report_results
from source.utils.readData import read_data
from source.utils.signal_preprocess import get_train_test_features
from source.utils.utils import *

ids = ['04010', '04011', '04012', '04013', '04014', '04015', '04016', '04017', '04018', '04019']
action = 'walk'
sensor = 'Left_foot'
N_ESTIMATORS_LIST = [25, 50, 100, 150, 200]
K_LIST = [3, 5, 7, 9, 11]


def get_safe_string(s: str):
    s = s.replace(' ', '_')
    return s


def get_classifiers():
    """
    Returns list of classifiers for this experiment
    :return: list of classifiers, and their titles
    """
    clsfr = [RandomForestClassifier(n_estimators=n) for n in N_ESTIMATORS_LIST]
    clsfr.extend([KNeighborsClassifier(n_neighbors=k) for k in K_LIST])
    clsfr.append(DecisionTreeClassifier())
    clsfr.append(SVC(kernel='rbf', C=0.05))
    clsfr.append(SVC(kernel='linear', C=0.05))
    clsfr_titles = ['RandomForest %d' % n for n in N_ESTIMATORS_LIST]
    clsfr_titles.extend('kNearestNeighbors %d' % k for k in K_LIST)
    clsfr_titles.append('DecisionTree default')
    clsfr_titles.append('SupportVector rbf')
    clsfr_titles.append('DecisionTree linear')
    return clsfr, clsfr_titles


if __name__ == '__main__':
    dataset = read_data(GAIT_DATA_HOME, ids, action, sensor)
    # 1. GAIT visualizing
    # task6(dataset[0])

    # 2. Train classifiers
    train_x, train_y, test_x, test_y = get_train_test_features(dataset)
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    classifiers, classifiers_title = get_classifiers()

    # Performing classification using each classifier
    for i in range(len(classifiers)):
        print('========================================================')
        print('\tRunning:', classifiers_title[i])
        print('========================================================')
        classifiers[i].fit(train_x, train_y)  # train
        pred_y = classifiers[i].predict(test_x)
        score, confusion_matrix, report = report_results(test_y, pred_y, cmat_flag=True)
        np.save(os.path.join(RESULTS_HOME, 'gait', get_safe_string(classifiers_title[i]) + '_cmat.npy'),
                confusion_matrix)
        print(score)
