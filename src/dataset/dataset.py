from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset,
)


import numpy as np


class AdapTrainDataset(Dataset):
    def __init__(self,
                 X_path = './dataset/train_x.npy',
                 y_path = './dataset/train_y.npy',):
        try:
            self.X = np.load(X_path)
            self.y = np.load(y_path)
        except FileNotFoundError:
            raise FileNotFoundError("Please put the dataset in the 'app/dataset' folder.")
        except Exception as e:
            raise e
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx].astype(np.float32), self.y[idx].astype(np.int64)