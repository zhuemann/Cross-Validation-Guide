from torch.utils.data import Dataset
import numpy as np
import torch
# This class takes in the data matrix and their labels and returns and
# 8x8 image casted as floats
class MNISTImageDataset(Dataset):
    def __init__(self, matrix, label):

        self.data = matrix
        self.target = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):

        image = np.reshape(self.data, (len(self.data), 8, 8))
        image = np.expand_dims(image, axis=1)

        return {
            'targets': torch.tensor(self.target[index], dtype=torch.float),
            'images': torch.tensor(image[index], dtype=torch.float)
        }