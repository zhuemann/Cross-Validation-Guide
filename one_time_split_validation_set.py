from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, LeavePOut
from setup.model import SimpleNet, test_model, train_model
from setup.MNISTImageDataset import MNISTImageDataset
from torch.utils.data import DataLoader
import numpy as np

# Split data into train, validation, and test splits which are stratified and then train and evaluate a model
def one_time_split_with_validation_set(data, targets):
    # One-time split with validation set
    # define how long we train for
    hyperparameters = [2, 5, 10]

    # defines the batch sizes and shuffles the data
    train_params = {'batch_size': 16,
                    'shuffle': True,
                    'num_workers': 1}

    # split data twice, the first time splits off the test set the second time splitting off the training and validation sets
    X_temp, X_test, Y_temp, Y_test = train_test_split(data, targets, test_size=0.3, stratify=targets, random_state=0)
    X_train, X_validate, Y_train, Y_validate = train_test_split(X_temp, Y_temp, test_size=20 / 70, stratify=Y_temp,
                                                                random_state=0)

    # use our data loader to shape the data
    training_set = MNISTImageDataset(X_train, Y_train)
    valid_set = MNISTImageDataset(X_validate, Y_validate)
    test_set = MNISTImageDataset(X_test, Y_test)

    # make all the loaders to be passed to the train and evaluate functions
    training_loader = DataLoader(training_set, **train_params)
    valid_loader = DataLoader(valid_set, **train_params)
    test_loader = DataLoader(test_set, **train_params)

    accuracies = []
    # loop through your different hyperparameters trying them on the valid set
    for parameter in hyperparameters:
        # train a model and get the best validation score and then evaluate that model on the test set
        best_model, valid_acc = train_model(n_epochs=parameter, train_loader=training_loader, valid_loader=valid_loader)
        accuracies.append(valid_acc)

    best_param = np.argmax(np.asarray(accuracies))
    best_model, valid_acc = train_model(n_epochs=hyperparameters[best_param], train_loader=training_loader,
                                        valid_loader=valid_loader)
    test_model(loader=test_loader, test_model=best_model)