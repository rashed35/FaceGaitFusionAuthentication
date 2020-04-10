"""
Created by Sanjay at 4/9/2020

Feature: Enter feature name here
Enter feature description here
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
from source.face.FisherFace import float2uint8

from source.utils.utils import *

plt.style.use('seaborn-white')


def task2(W_e, m):
    top8_faces = W_e[:, :8]
    fig = plt.figure(1)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(1, 10):
        t_img = float2uint8(top8_faces[:, i - 1:i].reshape(IMG_SHAPE)) if i < 9 else float2uint8(m.reshape(IMG_SHAPE))
        t_lbl = 'Eigenface ' + str(i) if i < 9 else 'mean'
        plt.subplot(3, 3, i)
        plt.title(t_lbl, color='#0000FF')
        plt.imshow(t_img)
    # plt.show()
    plt.savefig(os.path.join(RESULTS_HOME, 'task2_plots.png'), dpi=300)


def task3(k_list, accuracy_list, conf_mat_k5):
    plt.figure()
    for k, acc in zip(k_list, accuracy_list):
        plt.scatter(k, acc, color='red')
        plt.text(k+0.3, acc-0.005, 'k='+str(k)+'\nacc=%.2f' % acc, fontsize=9)
    plt.title('kNN classifier performance: k vs. accuracy')
    plt.xlim((2, 15))
    plt.xticks(np.arange(min(k_list)-1, max(k_list)+2, 1))
    plt.xlabel('k Neighbors')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(RESULTS_HOME, 'task3_k_vs_acc.png'), dpi=300)
    plt.show()
    plt.clf()

    df_cm = pd.DataFrame(conf_mat_k5, index=[i for i in range(10)], columns=[i for i in range(10)])
    plt.figure(figsize=(10, 7))
    plt.title('Confusion Matrix for kNN | k=5, D=30 (for PCA)')
    sn.heatmap(df_cm, annot=True)
    plt.savefig(os.path.join(RESULTS_HOME, 'task3_confusion_matrix_k5.png'), dpi=300)
    plt.show()


def task4(n_list, accuracy_list, conf_mat_n100):
    plt.figure()
    for n, acc in zip(n_list, accuracy_list):
        plt.scatter(n, acc, color='red')
        plt.text(n-0.3, acc-0.003, 'n=' + str(n) + '\nacc=%.2f' % acc, fontsize=9)
    plt.title('Random Forest performance: n estimators vs. accuracy')
    plt.xticks(n_list)
    plt.xlabel('n Estimators')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(RESULTS_HOME, 'task4_n_estimators_vs_acc.png'), dpi=300)
    plt.show()
    plt.clf()

    df_cm = pd.DataFrame(conf_mat_n100, index=[i for i in range(10)], columns=[i for i in range(10)])
    plt.figure(figsize=(10, 7))
    plt.title('Confusion Matrix for Random Forest | n=100, D=30 (for PCA)')
    sn.heatmap(df_cm, annot=True)
    plt.savefig(os.path.join(RESULTS_HOME, 'task4_confusion_matrix_n100.png'), dpi=300)
    plt.show()