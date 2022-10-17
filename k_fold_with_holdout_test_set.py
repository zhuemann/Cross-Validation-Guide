from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, LeavePOut
from setup.model import SimpleNet, test_model, train_model
from setup.MNISTImageDataset import MNISTImageDataset
from torch.utils.data import DataLoader
import numpy as np

# This stategy is very similar to the last one but we split off a test set before folding the remaining data.
# It will also bear a resemblance to nested cross validation. The biggest difference between the two is here we split
# a test set off and in nested cross validation we use another layer of cross validation.

def k_fold_with_holdout_test_set(data, targets):

    # defines the batch sizes and shuffles the data
    train_params = {'batch_size': 16,
                    'shuffle': True,
                    'num_workers': 1}

    # k fold with holdout will be the same thing as k fold expect we split the test set out at the start

    k_folds = 5

    # splits off the test set from the data that will be folded k times
    X_kfold, X_test, Y_kfold, Y_test = train_test_split(
        data, targets, test_size=0.3, random_state=42, shuffle=True, stratify=targets)

    # sets up the k folds of the data
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True)

    # here we use the number of epochs as the hyperparamter we wish to tune and will
    # select the number of epochs which does the best and retrain on all the data
    hyperparameters = [1, 5, 10]
    acc_matrix = np.zeros((k_folds, len(hyperparameters)))

    # loops through each fold up to k and trains a model and test on the kth fold
    for i, (train_index, test_index) in enumerate(skf.split(X_kfold, Y_kfold)):

        # these are the images used to train in the k folds
        train_image = X_kfold[train_index]
        train_targets = Y_kfold[train_index]

        # these are the k fold test images used for model selection
        valid_image = X_kfold[test_index]
        valid_targets = Y_kfold[test_index]

        # sets up our dataloader
        training_set = MNISTImageDataset(train_image, train_targets)
        valid_set = MNISTImageDataset(valid_image, valid_targets)

        training_loader = DataLoader(training_set, **train_params)
        valid_loader = DataLoader(valid_set, **train_params)

        for idx, parameter in enumerate(hyperparameters):
            best_model, _ = train_model(n_epochs=parameter, train_loader=training_loader)
            acc = test_model(loader=valid_loader, test_model=best_model)
            acc_matrix[i, idx] = acc

    # gets the bets parameter and prints it out for you
    fold_mean = np.mean(acc_matrix[:, :], axis=0)
    print(f"fold means: {fold_mean}")
    best_parameter = np.argmax(fold_mean)
    print(f"best_parameter_index: {best_parameter} , best_parameter: {hyperparameters[best_parameter]}")

    # sets up our dataloader with all of the data we used in the k folds
    training_set = MNISTImageDataset(X_kfold, Y_kfold)
    test_set = MNISTImageDataset(X_test, Y_test)
    training_loader = DataLoader(training_set, **train_params)
    test_loader = DataLoader(test_set, **train_params)

    # trains the model on all the data used in the k folds and test on the holdout test set
    best_model, _ = train_model(n_epochs=hyperparameters[best_parameter], train_loader=training_loader)
    acc = test_model(loader=test_loader, test_model=best_model)

    print(f"Final Accuracy on Holdout Test Set: = {acc}")