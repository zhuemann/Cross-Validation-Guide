from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, LeavePOut
from setup.model import SimpleNet, test_model, train_model
from setup.MNISTImageDataset import MNISTImageDataset
from torch.utils.data import DataLoader
import numpy as np

def k_folded_test_set(data, targets):

    # k fold
    k = 5
    epochs = 10
    # gets the indexes for each fold in our dataset
    skf = StratifiedKFold(n_splits=k, shuffle=True)

    acc_list = []

    # defines the batch sizes and shuffles the data
    train_params = {'batch_size': 16,
                    'shuffle': True,
                    'num_workers': 1}

    # loops through each fold up to k and trains a model and test on the kth fold
    for train_index, test_index in skf.split(data, targets):
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

        # trains the models and adds each evaluate accuracy to the list
        best_model, _ = train_model(n_epochs=epochs, train_loader=training_loader)
        acc = test_model(loader=test_loader, test_model=best_model)
        acc_list.append(acc)

    print(f"List of Accuracies For Each Fold: = {acc_list}")
    print(f"Average Fold Accuracy: = {np.mean(np.asarray(acc_list))}")



# takes as input the number of folds to split your data into and then your data
# and targets. It returns the index of each fold such that we can easily
# retrieve the data that belongs to each folder later on
def get_k_fold_index(folds = 1, fold_data = None, fold_targets = None):

  #this call does all the heavy lifting
  skf = StratifiedKFold(n_splits=folds, shuffle = True)

  train_fold = []
  test_fold = []
  for train_index, test_index in skf.split(fold_data, fold_targets):
    test_fold.append(test_index.tolist())
    train_fold.append(train_index.tolist())

  return train_fold, test_fold