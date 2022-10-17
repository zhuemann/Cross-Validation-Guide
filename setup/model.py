# A simple convolutional nueral network
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm


class SimpleNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(SimpleNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(8, 32, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# For each epoch of training we will train the model and then evaluate on the validation set if there is a
# validation set. Then we return the model which either performes best on the validation set or the model after
# training for specified number of epochs if no validation set is given.
def train_model(n_epochs = 5, train_loader = None, valid_loader = None, classes = 10):

    # create the inital model
    model = SimpleNet(num_classes = classes)

    # do some initial setup
    # makes sure that if you have a gpu availible you use it otherwise use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # defines the learning rate we will use for all the examples
    LR = .0005
    # defines the loss fucntion we will use
    criterion = torch.nn.CrossEntropyLoss()

    model.to(device)
    # defines which optimizer is being used
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

    best_valid_acc = -1
    best_model = model

    for epoch in range(0, n_epochs):
        #puts the model into train model so it updates weights
        model.train()

        fin_targets = []
        fin_outputs = []
        loss_array = []

        for _, training_samples in tqdm(enumerate(train_loader, 0)):
            # the label of the images in the batch
            targets = training_samples['targets'].to(device, dtype=torch.long)
            # the images in the batch
            images = training_samples['images'].to(device)

            # the prediction for the model
            outputs = model(images)

            # converts the targets and outputs to list to compute loss
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

            # gets the loss based on the model predictions
            loss = criterion(outputs,targets)
            loss_array.append(loss.cpu().detach().numpy().tolist())

            # zeros out the gradient from last step
            optimizer.zero_grad()

            # propagates loss to all the weights in the model
            loss.backward()

            # takes the step given the propagated loss and the optimzer
            optimizer.step()
        # should calculate average loss
        print(f"Epoch {str(epoch)}, Average Training Lose = {np.mean(np.asarray(loss_array))}")
        # if there is a validation loader we want to test our model on the
        # evaluation set and if it is better save it as the best model
        if valid_loader is not None:
            validation_acc = test_model(loader = valid_loader, test_model = model)
            if validation_acc > best_valid_acc:
                best_valid_acc = validation_acc
                best_model = model
            print(f"Epoch {str(epoch)}, Validation Score = {validation_acc}")

    return best_model, best_valid_acc


# Here we define the evaluation of the model on a test set in which we feed the test set to the model and check
# the outputs against the labels. In this step we don't update the model at all, it is just to estimate performance
# on new unseen data.
def test_model(loader=None, test_model=None):
    model = test_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # this should make sure the weights don't require a gradient
    model.eval()
    fin_targets = []
    fin_outputs = []
    # loops through all the batches of images and labels in our test set
    for _, test_samples in tqdm(enumerate(loader, 0)):
        # gets the batches of images and labels
        targets = test_samples['targets'].to(device, dtype=torch.long)
        images = test_samples['images'].to(device)
        # the model makes a prediction
        outputs = model(images)
        # the outputs are convered to lists appened onto out final prediction list
        fin_targets.extend(targets.cpu().detach().numpy().tolist())
        fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    # maybe cut this line
    final_outputs = np.copy(fin_outputs)
    # get the index of the best prediction
    final_outputs = np.argmax(final_outputs, axis=1)

    # gets the number of correct predictions from our final output
    num_correct = 0
    for i in range(0, len(fin_targets)):
        if (fin_targets[i] == final_outputs[i]):
            num_correct = num_correct + 1

    # calculates and returns the accuracy on the given test set
    accuracy = num_correct / len(fin_targets)
    print(f"Evaluation accuracy = {accuracy}")
    return accuracy