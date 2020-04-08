"""
Created by Sanjay at 4/8/2020

Feature: Enter feature name here
Enter feature description here
"""
import cv2

from source.FisherFace import read_faces, myPCA
from source.utils import *
import numpy as np

# ======================= SETTINGS ========================= #
D = 30  # Number of eigenfaces, for PCA
IMG_SHAPE = (160, 140)


def back_projection(f, W_e, do_save=False):
    """
    Back projects eigenfaces to 2D image
    :param f: face in PCA space (D dimension)
    :param W_e: Eigenfaces (D columns)
    :param do_save: flag to know if file should be saved
    :return: None
    """
    _x = np.dot(W_e, f) + m  # back projection of face
    _x = _x.reshape(IMG_SHAPE)  # reshape to 2D image
    if do_save:
        save_dir = os.path.join(PCA_BACK_PROJECT_DIR, 'D_' + str(D))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(PCA_BACK_PROJECT_DIR, 'D_' + str(D), str(labels[i]) + '.bmp'), _x)


if __name__ == '__main__':
    faces, labels = read_faces(FACE_TRAIN_DIR)  # read faces, labels
    W, LL, m = myPCA(faces)  # eigenfaces, eigenvalues, mean face
    W_e = W[:, :D]  # select only 'D' eigenfaces
    m = np.expand_dims(m, 1)  # (xxx,) -> (xxx, 1)

    n_faces = faces.shape[1]  # number of faces
    train_x = np.zeros(shape=(D, 0), dtype=float)
    for i in range(n_faces):
        f = np.dot(W_e.T, (faces[:, i:i + 1] - m))  # calculating PCA feature of single face
        train_x = np.hstack((train_x, f))

    print(train_x.shape, _x.shape)
    cv2.imshow('', faces[:, 0:1].reshape((160, 140)).astype('uint8'))
    cv2.imshow('face image', _x.astype('uint8'))
    cv2.waitKey();
