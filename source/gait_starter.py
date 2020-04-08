from utils.readData import read_data
import os

from utils.signal_preprocess import get_train_test_features

ids = ['04010', '04011', '04012', '04013', '04014', '04015', '04016', '04017', '04018', '04019']
action = 'walk'
sensor = 'Left_foot'
dirname = os.path.dirname(__file__)
dataset_dir = os.path.join(dirname, 'GAIT')

# reads in GAIT data from files. Make sure the unzipped GAIT folder is inside the working directory
dataset = read_data(dataset_dir, ids, action, sensor)

# this dataset contains the gait signals for the 10 IDs.
# dataset[0] is the signal for person ID 04010
# dataset[0] contains 1801 x,y,z time stamp values for this person
# Use this and write your code to visualize the signal of ID-04010

# ToDO: 1. GAIT visualizing

# pass this dataset to get the features and to split them to train and test sets
train_x, train_y, test_x, test_y = get_train_test_features(dataset)

# use this train and test sets to evaluate classifiers for GAIT

# ToDo: 2. Train classifiers
