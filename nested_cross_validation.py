from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, LeavePOut
from setup.model import SimpleNet, test_model, train_model
from setup.MNISTImageDataset import MNISTImageDataset
from torch.utils.data import DataLoader
import numpy as np

# Nested cross validation is similar to the K-fold with holdout test set expect instead of splitting off a holdout
# test at the beginning and use that to evaluate we are going to use folds of the data where we use each fold once
# to evaluate the model selected by the inner loop. Then we average this outer loop set of accuries to get the final
# perforamnce.
def nested_cross_validation(data, targets):

    # defines the batch sizes and shuffles the data
    train_params = {'batch_size': 16,
                    'shuffle': True,
                    'num_workers': 1}

    # Nested Cross Validatoin!!!

    hyperparameters = [1, 5, 10]

    k_outer_folds = 5
    k_inner_folds = 4

    acc_matrix = np.zeros((k_outer_folds, k_inner_folds, len(hyperparameters)))

    # sets up the k folds of the data
    skf_outer = StratifiedKFold(n_splits=k_outer_folds, shuffle=True)
    skf_inner = StratifiedKFold(n_splits=k_inner_folds, shuffle=True)

    outer_acc = []

    # Outer loop for performance estimation
    for k, (outer_train_index, outer_test_index) in enumerate(skf_outer.split(data, targets)):
        # these are the images used in the k folds
        X_inner = data[outer_train_index]
        Y_inner = targets[outer_train_index]

        # these are the k fold test images used for model selection
        X_test = data[outer_test_index]
        Y_test = targets[outer_test_index]

        outer_train_set = MNISTImageDataset(X_inner, Y_inner)
        test_set = MNISTImageDataset(X_test, Y_test)

        # inner loop for model selection
        # for j in range(0,k_folds - 1):
        for j, (inner_train_index, inner_valid_index) in enumerate(skf_inner.split(X_inner, Y_inner)):

            # sets up the training and valid loader
            X_train_inner = X_inner[inner_train_index]
            Y_train_inner = Y_inner[inner_train_index]
            X_valid_inner = X_inner[inner_valid_index]
            Y_valid_inner = Y_inner[inner_valid_index]

            training_set = MNISTImageDataset(X_train_inner, Y_train_inner)
            valid_set = MNISTImageDataset(X_valid_inner, Y_valid_inner)
            training_loader = DataLoader(training_set, **train_params)
            valid_loader = DataLoader(valid_set, **train_params)

            # for each of the models or hyperparamters you want to try train and evaluate
            for epoch_idx, param in enumerate(hyperparameters):
                best_model, _ = train_model(n_epochs=param, train_loader=training_loader)
                acc = test_model(loader=valid_loader, test_model=best_model)
                acc_matrix[k, j, epoch_idx] = acc

        outer_training_loader = DataLoader(outer_train_set, **train_params)
        outer_test_loader = DataLoader(test_set, **train_params)

        # calculates the mean of the models for each fold given the parameter your testing
        fold_mean = np.mean(acc_matrix[k, :, :], axis=0)
        # picks out the index of the best parameter/model
        best_parameter = np.argmax(fold_mean)

        # use that best parameter to train a new model on all of the outer fold data
        best_model, _ = train_model(n_epochs=hyperparameters[best_parameter], train_loader=outer_training_loader)
        # test this best model on the outer fold hold out fold and add it to the list of accuracies
        acc = test_model(loader=outer_test_loader, test_model=best_model)

        outer_acc.append(acc)

    outer_acc = np.array(outer_acc)
    print(f"Average of outer loop folds: = {np.mean(outer_acc)}")
    print(f"Standard deviation of fold accuracies: = {np.std(outer_acc)}")