"""
Created by Sanjay at 4/8/2020

Feature: Enter feature name here
Enter feature description here
"""
import os

PROJECT_HOME = r'C:\Users\Sanjay Saha\CS5332Assignment2'
DATASET_HOME = os.path.join(PROJECT_HOME, 'data')
RESULTS_HOME = os.path.join(PROJECT_HOME, 'results')

FACE_DATA_HOME = os.path.join(DATASET_HOME, 'face')
FACE_TRAIN_DIR = os.path.join(FACE_DATA_HOME, 'train')
FACE_TEST_DIR = os.path.join(FACE_DATA_HOME, 'test')

GAIT_DATA_HOME = os.path.join(DATASET_HOME, 'gait')

PCA_BACK_PROJECT_DIR = os.path.join(RESULTS_HOME, 'pca_back_projection')
