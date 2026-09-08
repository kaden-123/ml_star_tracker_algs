import torch
from torch import nn
import h5py
from torch.utils.data import Dataset

class star_tracker_v1(nn.Module):
    def __init__(self, n_bins, n_classes, hidden_one, hidden_two):
        super().__init__()

        self.bn1 = nn.BatchNorm1d(n_bins) 
        
        self.grp1 = nn.Sequential(
            nn.Linear(n_bins, hidden_one),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_one),
            nn.Dropout(p=0.2),
        )

        self.grp2 = nn.Sequential(
            nn.Linear(hidden_one, hidden_two),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_two),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_two, n_classes)
        )
        
    def forward(self, x): 
        x = self.bn1(x)
        x = self.grp1(x)
        x = self.grp2(x)
        return x

class H5Data(Dataset):
    def __init__(self, path):
        self.path = path
        self.index_map = []
        self.file = h5py.File(path, "r")
        for group in self.file.keys():
            for sample in self.file[str(group)]:
                self.index_map.append((group, sample))
                    
        self.length = len(self.index_map)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        group, sample = self.index_map[idx]
        data = self.file[group][sample][:]

        return torch.tensor(data, dtype=torch.float32), int(float(group))