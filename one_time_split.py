from setup.model import SimpleNet, test_model, train_model
from setup.MNISTImageDataset import MNISTImageDataset
from torch.utils.data import DataLoader


from sklearn.model_selection import train_test_split

# Here we split the data into a training set and test set. Then train our model and evalute on the test set.
def one_time_split(data, targets):

    # defines the batch sizes and shuffles the data
    train_params = {'batch_size': 16,
                    'shuffle': True,
                    'num_workers': 1}

    # splits the data and stratifies it according to the targets
    train_data, test_data, train_targets, test_targets = train_test_split(
        data, targets, test_size=0.2, random_state=42, shuffle=True, stratify=targets)

    # sets up the dataloaders
    training_set = MNISTImageDataset(train_data, train_targets)
    test_set = MNISTImageDataset(test_data, test_targets)

    training_loader = DataLoader(training_set, **train_params)
    test_loader = DataLoader(test_set, **train_params)

    # train the model
    best_model, valid_acc = train_model(n_epochs=5, train_loader=training_loader, valid_loader=None)

    # test the model
    final_accuracy = test_model(loader=test_loader, test_model=best_model)
    print(f"Final Accuracy: = {final_accuracy}")