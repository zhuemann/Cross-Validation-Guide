from setup.model import SimpleNet, test_model, train_model
from setup.MNISTImageDataset import MNISTImageDataset
from one_time_split import one_time_split
from one_time_split_validation_set import one_time_split_with_validation_set

import torch
from sklearn.datasets import load_digits

if __name__ == '__main__':


    # load in the data from above
    digits = load_digits()
    data = digits.data
    targets = digits.target

    #one_time_split(data, targets)

    one_time_split_with_validation_set(data, targets)