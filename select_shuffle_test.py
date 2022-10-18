from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, LeavePOut
from setup.model import SimpleNet, test_model, train_model
from setup.MNISTImageDataset import MNISTImageDataset
from torch.utils.data import DataLoader
import numpy as np


def select_shuffle_test(data, targets):

    # defines the batch sizes and shuffles the data
    train_params = {'batch_size': 16,
                    'shuffle': True,
                    'num_workers': 1}

    # Select Shuffle Test
    hyperparameters = [1, 5, 10]

    k_fold_selection = 5

    acc_matrix = np.zeros((k_fold_selection, len(hyperparameters)))

    # sets up the k folds of the data
    skf_select = StratifiedKFold(n_splits=k_fold_selection, shuffle=True)

    # loop through all of the folds of the data
    for j, (inner_train_index, inner_valid_index) in enumerate(skf_select.split(data, targets)):

        # sets up the training and valid loader
        X_train_inner = data[inner_train_index]
        Y_train_inner = targets[inner_train_index]
        X_valid_inner = data[inner_valid_index]
        Y_valid_inner = targets[inner_valid_index]

        training_set = MNISTImageDataset(X_train_inner, Y_train_inner)
        valid_set = MNISTImageDataset(X_valid_inner, Y_valid_inner)
        training_loader = DataLoader(training_set, **train_params)
        valid_loader = DataLoader(valid_set, **train_params)

        # for each of the models or hyperparamters you want to try train and evaluate
        for epoch_idx, param in enumerate(hyperparameters):
            best_model, _ = train_model(n_epochs=param, train_loader=training_loader)
            acc = test_model(loader=valid_loader, test_model=best_model)
            acc_matrix[j, epoch_idx] = acc

    print(acc_matrix)

    # calculates the mean of the models for each fold given the parameter your testing
    fold_mean = np.mean(acc_matrix[:, :], axis=0)
    # picks out the index of the best parameter/model
    best_parameter = np.argmax(fold_mean)
    print(f"Selected hyperparameter index: {best_parameter} for {hyperparameters[best_parameter]} total epochs")

    # k fold
    k_fold_test = 5
    skf_test = StratifiedKFold(n_splits=k_fold_test, shuffle=True)

    acc_list = []

    # loops through each fold up to k and trains a model and test on the kth fold
    for train_index, test_index in skf_test.split(data, targets):
        # gets data from our largers array for k-1 folds
        X_train = data[train_index]
        Y_train = targets[train_index]
        # gets the test set
        X_test = data[test_index]
        Y_test = targets[test_index]

        # sets up our data loaders
        training_set = MNISTImageDataset(X_train, Y_train)
        test_set = MNISTImageDataset(X_test, Y_test)
        training_loader = DataLoader(training_set, **train_params)
        test_loader = DataLoader(test_set, **train_params)

        # use that best parameter to train a new model on all of the outer fold data
        best_model, _ = train_model(n_epochs=hyperparameters[best_parameter], train_loader=training_loader)
        # test this best hyperparamter set on the shuffled data
        acc = test_model(loader=test_loader, test_model=best_model)
        acc_list.append(acc)

    print(f"List of Accuracies For Each Fold: = {acc_list}")
    print(f"Average Fold Accuracy: = {np.mean(np.asarray(acc_list))}")