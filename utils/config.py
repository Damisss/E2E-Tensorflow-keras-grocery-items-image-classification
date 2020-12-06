import numpy as np
import os

BATCH_SIZE=16
EPOCHS = 40

MEAN = np.array([123.68, 116.779, 103.939], dtype=np.float32)

CLASSES = {
   'Apple': 0,
    'Banana': 1,
    'Melon': 2,
    'Juice': 3,
    'Milk': 4,
    'Ginger': 5,
    'Pepper': 6,
    'Tomato': 7,
    'Lemon': 8,
    'Avocado':9,
    'Kiwi': 10
}
# path
ORG_DATASET_PATH = os.path.sep.join(['dataset', 'GroceryStoreDataset', 'dataset'])

# basePath for new dataset
BASE_PATH = 'dataset'
PLOT_PATH = 'output/training.png'


# training data
X_TRAIN = 'dataset/X_train.pickle'
Y_TRAIN = 'dataset/y_train.pickle'
X_TEST = 'dataset/X_test.pickle'
Y_TEST = 'dataset/y_test.pickle'
X_VAL = 'dataset/X_val.pickle'
Y_VAL = 'dataset/y_val.pickle'

# LabelBinarizer
LB = 'output/le.pickle'
