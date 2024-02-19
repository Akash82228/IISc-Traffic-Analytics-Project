import torch
import numpy as np
import pandas as pd
import os
from torch_geometric.data import InMemoryDataset, Data

class SumoTrafficDataset(InMemoryDataset):
    """
    Dataset for Graph Neural Networks applied to traffic prediction with the SUMO dataset.
    """
    def __init__(self, root, window, horizon, name_scaler="max01", transform=None, pre_transform=None):

        self.window = window
        self.horizon = horizon
        self.name_scaler = name_scaler

        super().__init__(root, transform, pre_transform)
        self.data, self.slices, self.n_node, self.scaler = torch.load(self.processed_paths[0])
        
    
    @property
    def raw_file_names(self):
        return ['real_data.csv', 'FINAL_EDGE_LIST.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):

        # print(f"window: {self.window}, horizon: {self.horizon}")  # NOTE:DEBUG

        vehicle_data = pd.read_csv(os.path.join(self.root, 'real_data.csv'))
        edges = pd.read_csv(os.path.join(self.root, 'FINAL_EDGE_LIST.csv'))

        # Prepare edge data
        node_mapping = {node: idx for idx, node in enumerate(vehicle_data['node_ID'].unique())}
        edge_index = torch.tensor([[node_mapping[src], node_mapping[dst]] for src, dst in zip(edges['source'], edges['target'])], dtype=torch.long).t()

        # Assign weight of 1 to each edge
        edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float)

        # Prepare node features and labels
        features = vehicle_data.drop(columns=['node_ID']).to_numpy().T  # Time steps are columns now, nodes are rows

        # Normalize features using the provided scaler
        normalized_features, scaler = normalize_dataset(features, self.name_scaler)

        # Creating PyG data objects for each time step that can be fully featurized
        data_list = []
        for time_step in range(normalized_features.shape[0] - (self.window + self.horizon) + 1):
            x = torch.tensor(normalized_features[time_step:time_step+ self.window], dtype=torch.float).t()
            y = torch.tensor(normalized_features[time_step + self.window:time_step + self.window + self.horizon], dtype=torch.float).t()
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(data)

        self.n_node = len(node_mapping)  # Set number of nodes
        data, slices = self.collate(data_list)
        torch.save((data, slices, self.n_node, scaler), self.processed_paths[0])
        self.scaler = scaler  # Save scaler to the object


# data normalization
# ***********************************************************************************
class NScaler(object):
    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

class StandardScaler(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return data * self.std + self.mean


class MinMax01Scaler(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return data * (self.max - self.min) + self.min


class MinMax11Scaler(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min) * 2. - 1.

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return  ((data + 1.) / 2.) * (self.max - self.min) + self.min


def normalize_dataset(data, normalizer):
    if normalizer == 'max01':
        minimum = data.min()
        maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print("Normalize the dataset by MinMax01 Normalization", minimum, maximum)
    elif normalizer == 'max11':
        minimum = data.min()
        maximum = data.max()
        scaler = MinMax11Scaler(min= minimum, max=maximum)
        data = scaler.transform(data)
        print("Normalize the dataset by MinMax11 Normalization", minimum, maximum)
    elif normalizer == 'std':
        mean = data.mean()
        std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print("Normalize the dataset by StandardScaler Normalization", mean, std)
    elif normalizer == None:
        scaler = NScaler()
        data = scaler.transform(data)
        print("Does not normalize the dataset")
    else:
        raise ValueError
    return data, scaler
# ***********************************************************************************
