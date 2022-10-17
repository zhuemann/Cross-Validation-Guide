from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, LeavePOut
from setup.model import SimpleNet, test_model, train_model
from setup.MNISTImageDataset import MNISTImageDataset
from torch.utils.data import DataLoader
import numpy as np


#In leave-one-out you leave out one sample from your dataset which you will use as your test set, then you train
# on all the other samples. You do this for all of the samples in your data set leaving each one out a single time,
# you then average the accuracy over all of the left out samples. In this example we are going to limit our data to
# binary classification of 0's and 1's for only. In this example we grab the first 100 samples from out dataset and
# use the 0's and 1's as our dataset.
def leave_one_out(data, targets):

    # defines the batch sizes and shuffles the data
    train_params = {'batch_size': 16,
                    'shuffle': True,
                    'num_workers': 1}

    # creates the binary dataset which we will use for leave-p-out
    binary_data = []
    binary_targets = []
    for i in range(0, 100):
        if targets[i] == 0:
            binary_data.append(data[i])
            binary_targets.append(targets[i])
        if targets[i] == 1:
            binary_data.append(data[i])
            binary_targets.append(targets[i])

    binary_data = np.array(binary_data)
    binary_targets = np.array(binary_targets)

    p = 1
    n = len(binary_data)
    lpo = LeavePOut(p)
    split_num = lpo.get_n_splits(data[0:n])
    print(f"number of splits: {split_num}")

    acc_list = []
    epoch = 10

    for train_index, test_index in lpo.split(binary_data[0:n]):
        # uses the indexes defined by the lpo split to get those data points
        X_train, X_valid = binary_data[train_index], binary_data[test_index]
        Y_train, Y_valid = binary_targets[train_index], binary_targets[test_index]

        # loads them into the datalaoders
        training_set = MNISTImageDataset(X_train, Y_train)
        test_set = MNISTImageDataset(X_valid, Y_valid)

        training_loader = DataLoader(training_set, **train_params)
        test_loader = DataLoader(test_set, **train_params)

        # trains and tests the data
        best_model, _ = train_model(n_epochs=epoch, train_loader=training_loader, classes=2)
        acc = test_model(loader=test_loader, test_model=best_model)
        acc_list.append(acc)

    print(f"Average Leave-One-Out Accuracy: = {np.mean(np.asarray(acc_list))}")