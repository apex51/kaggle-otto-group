'''this file contains:

a 2-level stacking
- xgb in 30 runs
- nn in 30 runs
author: apex

'''
import numpy as np
from model.level1 import *
from model.level2 import *
from auxiliary import *

if __name__ == '__main__':

    # set the l1_train/l2_train split ratio
    train_split = 30000
    train_x, train_y, test_x = load_data('data/test.csv', 'data/test.csv', train_split)

    num_classes = len(np.unique(train_y))
    num_tests = test_x.shape[0]

    pred1 = np.zeros((num_tests, num_classes)).astype(np.float32)
    pred2 = np.zeros((num_tests, num_classes)).astype(np.float32)

    # level in 30 runs
    for i in range(30):
        pred1 += xgb_level2(train_x, train_y, test_x)
        pred2 += nn_level2(train_x, train_y, test_x)

    # combine by averaging
    pred1 = pred1/30
    pred2 = pred2/30
    pred = (pred1 + pred2)/2
    
    submit(pred)