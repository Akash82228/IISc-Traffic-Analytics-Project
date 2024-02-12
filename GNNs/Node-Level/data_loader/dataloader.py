import torch
import numpy as np
import pandas as pd
import os
from torch_geometric.data import InMemoryDataset, Data

class SumoTrafficDataset(InMemoryDataset):
    """
    Dataset for Graph Neural Networks applied to traffic prediction with the SUMO dataset.
    """
    def __init__(self, root, hist, fut, transform=None, pre_transform=None):

        self.hist = hist
        self.fut = fut
        super().__init__(root, transform, pre_transform)
        self.data, self.slices, self.n_node = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return ['real_data.csv', 'FINAL_EDGE_LIST.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):

        # print(f"hist: {self.hist}, fut: {self.fut}")  # NOTE:DEBUG

        vehicle_data = pd.read_csv(os.path.join(self.root, 'real_data.csv'))
        edges = pd.read_csv(os.path.join(self.root, 'FINAL_EDGE_LIST.csv'))

        # Prepare edge data
        node_mapping = {node: idx for idx, node in enumerate(vehicle_data['node_ID'].unique())}
        edge_index = torch.tensor([[node_mapping[src], node_mapping[dst]] for src, dst in zip(edges['source'], edges['target'])], dtype=torch.long).t()

        # Assign weight of 1 to each edge
        edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float)

        # Prepare node features and labels
        features = vehicle_data.drop(columns=['node_ID']).to_numpy().T  # Time steps are columns now, nodes are rows

        # Normalize features
        # self.mean = np.mean(features)
        # self.std_dev = np.std(features)
        # print(f"Mean of the dataset: {self.mean}")
        # print(f"Std Deviation of the dataset: {self.std_dev}")
        # normalized_features = (features - self.mean) / self.std_dev

        # print(normalized_features.shape[0])
        # print(normalized_features.shape[0] - (60 + 15))
        # print(range(normalized_features.shape[0] - (60 + 15)))

        self.min_val = np.min(features)
        self.max_val = np.max(features)
        normalized_features = (features - self.min_val) / (self.max_val - self.min_val)

        print(self.min_val)
        print(self.max_val)

        # Creating PyG data objects for each time step that can be fully featurized
        data_list = []
        for time_step in range(normalized_features.shape[0] - (self.hist + self.fut) + 1):
            x = torch.tensor(normalized_features[time_step:time_step+ self.hist], dtype=torch.float).t()
            y = torch.tensor(normalized_features[time_step + self.hist:time_step + self.hist + self.fut], dtype=torch.float).t()
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(data)

        self.n_node = len(node_mapping)  # Set number of nodes
        data, slices = self.collate(data_list)
        torch.save((data, slices, self.n_node), self.processed_paths[0])

