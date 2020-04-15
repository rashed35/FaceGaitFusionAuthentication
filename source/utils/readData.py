import os
import numpy as np


def read_data(dir, index_values, action, sensor):
    input_ids = []
    dataset = []
    for subject_id in index_values:
        data_dir = os.path.join(dir, subject_id, action, sensor + '.csv')
        data_x = np.genfromtxt(data_dir, delimiter=',', skip_header=True)  # Read from CSV
        data_x = data_x[:, 1:]  # Drop 1st column: timestamp

        data = data_x
        # data
        data = np.nan_to_num(data)  # Drop the 1st row which was empty (nan)
        input_ids.append(subject_id)
        dataset.append(data)
    return dataset
