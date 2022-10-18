from setup.model import SimpleNet, test_model, train_model
from setup.MNISTImageDataset import MNISTImageDataset
from one_time_split import one_time_split
from one_time_split_validation_set import one_time_split_with_validation_set
from k_folded_test_set import k_folded_test_set
from k_fold_with_holdout_test_set import k_fold_with_holdout_test_set
from leave_one_out import leave_one_out
from nested_cross_validation import nested_cross_validation
from monte_carlo_cross_validation import monte_carlo_cross_validation
from select_shuffle_test import select_shuffle_test

import torch
from sklearn.datasets import load_digits

if __name__ == '__main__':


    # load in the data from above
    digits = load_digits()
    data = digits.data
    targets = digits.target

    #one_time_split(data, targets)

    #one_time_split_with_validation_set(data, targets)

    k_folded_test_set(data, targets)
    k_fold_with_holdout_test_set(data, targets)
    leave_one_out(data, targets)
    nested_cross_validation(data, targets)
    select_shuffle_test(data, targets)
    monte_carlo_cross_validation(data, targets)