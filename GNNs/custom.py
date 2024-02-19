import os
import torch
import numpy as np
import pandas as pd
import warnings
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

class CustomTrafficDataset(Dataset):
    def __init__(self, data, window, horizon, single=False):
        self.window = window
        self.horizon = horizon
        self.single = single
        self.data = self._generate_windows(data)
    
    def _generate_windows(self, data):
        length = len(data)
        end_index = length - self.horizon - self.window + 1
        if end_index <= 0:
            return np.empty((0, self.window, data.shape[1], data.shape[2])), np.empty((0, self.horizon, data.shape[1], data.shape[2]))
        X, Y = [], []
        for index in range(end_index):
            X.append(data[index : index + self.window])
            if self.single:
                Y.append(data[index + self.window + self.horizon - 1 : index + self.window + self.horizon])
            else:
                Y.append(data[index + self.window : index + self.window + self.horizon])
        X = np.array(X)
        Y = np.array(Y)
        return X, Y
    
    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index]

def load_custom_dataset(real_data_path, input_dim=1):
    real_data_df = pd.read_csv(real_data_path)
    data = real_data_df.iloc[:, 1:].values
    data = data[..., np.newaxis]  # Add dimension for feature
    print("Loaded dataset shaped: ", data.shape)
    return data

def normalize_dataset(data, normalizer='std'):
    if normalizer == 'std':
        mean = np.mean(data, axis=(0, 1, 2), keepdims=True)
        std = np.std(data, axis=(0, 1, 2), keepdims=True)
        data = (data - mean) / std
        scaler = (mean, std)
    elif normalizer == 'minmax':
        min_val = np.min(data, axis=(0, 1, 2), keepdims=True)
        max_val = np.max(data, axis=(0, 1, 2), keepdims=True)
        data = (data - min_val) / (max_val - min_val)
        scaler = (min_val, max_val)
    else:
        raise ValueError("Unknown normalizer")
    return data, scaler

def split_data_by_ratio(data, val_ratio, test_ratio):
    len_data = data.shape[0]
    test_len = int(len_data * test_ratio)
    val_len = int(len_data * val_ratio)
    train_len = len_data - test_len - val_len
    
    if train_len <= 0 or val_len <= 0 or test_len <= 0:
        raise ValueError("The dataset split resulted in a non-positive length for one of the splits.")
    
    test_data = data[-test_len:]
    val_data = data[-(test_len + val_len):-test_len]
    train_data = data[:-(test_len + val_len)]
    
    return train_data, val_data, test_data

def get_dataloader(args, normalizer='std', single=False):
    data = load_custom_dataset(args.real_data_path, args.input_dim)
    data, scaler = normalize_dataset(data, normalizer)
    
    train_data, val_data, test_data = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    
    print(f"train_data: {train_data.shape}")
    print(f"val_data: {val_data.shape}")  
    print(f"test_data: {test_data.shape}")

    train_dataset = CustomTrafficDataset(train_data, args.window, args.horizon, single)
    val_dataset = CustomTrafficDataset(val_data, args.window, args.horizon, single)
    test_dataset = CustomTrafficDataset(test_data, args.window, args.horizon, single)

    print(f"train_dataset: {train_dataset}")
    print(f"val_dataset: {val_dataset}")
    print(f"test_dataset: {test_dataset}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader, scaler

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Custom Traffic Data Loader')
    parser.add_argument('--real_data_path', default='/home/akashs/codes/IISc-Traffic-Analytics-Project/GNNs/Node-Level/sumo_dataset/real_data.csv', type=str)
    parser.add_argument('--input_dim', default=1, type=int)
    parser.add_argument('--val_ratio', default=0.2, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--window', default=12, type=int)
    parser.add_argument('--horizon', default=12, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()
    
    train_loader, val_loader, test_loader, scaler = get_dataloader(args, normalizer='std', single=False)
    
    for x_batch, y_batch in train_loader:
        print(x_batch.shape, y_batch.shape)
        break
