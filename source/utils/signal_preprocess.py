import numpy as np
import math
import os
from scipy.signal import find_peaks
from source.utils import loess
import statistics
from scipy.stats import kurtosis, skew


def get_priod_crops(signal, person):
    # takes the signal for one person and the person ID as input. crops the signal
    # into periods and returns the crops as a sqrt signal, x,y,z signals.

    signal_squared = np.square(signal)
    signal_summed = np.sum(signal_squared, axis=1)
    signal_sqrt = np.sqrt(signal_summed)

    signalx = signal[:, 0]
    signaly = signal[:, 1]
    signalz = signal[:, 2]
    signal = signalx
    # signal = signal_sqrt

    prom = 0.8
    if int(person) == 33:
        prom = 0.9
    if int(person) == 36 or int(person) == 50 or int(person) == 53 or int(person) == 51 or int(person) == 4:
        prom = 0.7

    x = loess.loessline(signal)
    x = x[100: -100]

    peaks, _ = find_peaks(x, prominence=(prom, None))
    vallies, _ = find_peaks(x * -1, prominence=(prom, None))

    total_dist = 0
    dists = []
    for i in range(0, vallies.__len__() - 1):
        dist = vallies[i + 1] - vallies[i]
        total_dist += dist
        dists.append(dist)
    if dists.__len__() != 0:
        median_dist = statistics.median(dists)

        cropped_instances = []
        cropped_instances_x = []
        cropped_instances_y = []
        cropped_instances_z = []
        crop_start = []
        crop_end = []
        for i in range(0, vallies.__len__() - 1):
            dist = vallies[i + 1] - vallies[i]
            if abs(dist - median_dist) < median_dist / 3:
                cropped_instances.append(signal_sqrt[vallies[i]: vallies[i + 1]])
                cropped_instances_x.append(signalx[vallies[i]: vallies[i + 1]])
                cropped_instances_y.append(signaly[vallies[i]: vallies[i + 1]])
                cropped_instances_z.append(signalz[vallies[i]: vallies[i + 1]])
                crop_start.append(vallies[i])
                crop_end.append(vallies[i + 1])

        # print(cropped_instances.__len__(), " crops detected for p =", person)

        return cropped_instances, cropped_instances_x, cropped_instances_y, cropped_instances_z

    else:
        print("No crops detected for p=", person)


def period_based_features(crops):
    # takes in an array of period crops (one axis), returns set of features
    features = []
    for crop in crops:
        data_window = np.array(crop)

        acf = np.correlate(data_window, data_window, mode='full')
        acv = np.cov(data_window.T, data_window.T)
        sq_err = (data_window - np.mean(data_window)) ** 2

        features.append([np.max(data_window),  # 00
                         np.min(data_window),  # 01
                         np.size(data_window),  # 02
                         np.mean(data_window),  # 03
                         np.std(data_window),  # 04
                         np.var(data_window),  # 05
                         np.mean(acf),  # 06
                         np.std(acf),  # 07
                         np.mean(acv),  # 08
                         np.std(acv),  # 09
                         skew(data_window),  # 10
                         kurtosis(data_window),  # 11
                         math.sqrt(np.mean(sq_err))])  # 12

    return features


def get_train_test_features(dataset):
    featureset_train = np.zeros(shape=(0, 40), dtype=float)
    featureset_test = np.zeros(shape=(0, 40), dtype=float)
    for idx, signal in enumerate(dataset):
        crops = get_priod_crops(signal, idx + 1)
        num_crops = crops[1].__len__()
        # print(idx, num_crops)
        if num_crops >= 16:
            curr_label = idx + 1  # select the user ID as the label
            label_train = np.array([curr_label] * 8).reshape(8, 1)  # label array with same label for all crops
            label_test = np.array([curr_label] * 8).reshape(8, 1)  # label array with same label for all crops

            features_train = np.hstack((period_based_features(crops[1][0:8]), period_based_features(crops[2][0:8]),
                                        period_based_features(crops[3][0:8]), label_train))

            features_test = np.hstack((period_based_features(crops[1][8:16]), period_based_features(crops[2][8:16]),
                                       period_based_features(crops[3][8:16]), label_test))

            featureset_train = np.vstack((featureset_train, features_train))
            featureset_test = np.vstack((featureset_test, features_test))
        else:
            print("error in number of crops")

    train_x = featureset_train[:, 0:39]
    train_y = featureset_train[:, 39]
    test_x = featureset_test[:, 0:39]
    test_y = featureset_test[:, 39]

    return train_x, train_y, test_x, test_y
