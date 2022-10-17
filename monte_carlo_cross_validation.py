from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, LeavePOut
from setup.model import SimpleNet, test_model, train_model
from setup.MNISTImageDataset import MNISTImageDataset
from torch.utils.data import DataLoader
import numpy as np

def monte_carlo_cross_validation(data, targets):

    # defines the batch sizes and shuffles the data
    train_params = {'batch_size': 16,
                    'shuffle': True,
                    'num_workers': 1}

    # Monte Carlo
    # Set up seeds and array to store accuracies
    seed_list = [42, 151, 297, 333, 406]
    acc_array = []

    for seed in seed_list:
        # split your data into train, validate and test sets
        X_temp, X_test, Y_temp, Y_test = train_test_split(data, targets, test_size=0.3, stratify=targets,
                                                          random_state=seed)
        X_train, X_validate, Y_train, Y_validate = train_test_split(X_temp, Y_temp, test_size=20 / 70, stratify=Y_temp,
                                                                    random_state=seed)

        training_set = MNISTImageDataset(X_train, Y_train)
        valid_set = MNISTImageDataset(X_validate, Y_validate)
        test_set = MNISTImageDataset(X_test, Y_test)

        training_loader = DataLoader(training_set, **train_params)
        valid_loader = DataLoader(valid_set, **train_params)
        test_loader = DataLoader(test_set, **train_params)

        best_model, _ = train_model(n_epochs=10, train_loader=training_loader, valid_loader=valid_loader)
        acc = test_model(loader=test_loader, test_model=best_model)
        acc_array.append(acc)

    print(f"Average of all seeds: = {np.array(np.mean(acc_array))}")
    print(f"Standard deviation of all seeds: = {np.array(np.std(acc_array))}")