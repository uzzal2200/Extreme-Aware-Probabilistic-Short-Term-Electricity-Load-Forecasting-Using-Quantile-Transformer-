import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class ERCOTDataset(Dataset):
    def  __init__(self, data_file, window=24*7, normalize=True):
        super().__init__()
        self.data = pd.read_csv(data_file)
        
        if normalize:
            self.mean = np.mean(self.data)
            self.std = np.std(self.data)
            self.data = (self.data - self.mean) / self.std
        else:
            self.mean = 0
            self.std = 1
            
        self.window = window
        
    def  __len__(self):
        return len(self.data) - self.window
    
    def  __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.window], dtype=torch.float32)
        y_max = self.data.iloc[idx+self.window-1]['system demand']   # Max of the window
        extreme = y_max > np.percentile(self.data[:idx+self.window]['system demand'], 95)
        
        return x, (y_max, extreme), self.mean, self.std