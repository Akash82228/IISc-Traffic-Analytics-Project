import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv, MLP
from torch_geometric_temporal.nn.recurrent import A3TGCN
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ST_GAT(torch.nn.Module):
    """
    Spatio-Temporal Graph Attention Network as presented in https://ieeexplore.ieee.org/document/8903252
    """
    def __init__(self, in_channels, out_channels, n_nodes, heads=8, dropout=0.0):
        """
        Initialize the ST-GAT model
        :param in_channels Number of input channels
        :param out_channels Number of output channels
        :param n_nodes Number of nodes in the graph
        :param heads Number of attention heads to use in graph
        :param dropout Dropout probability on output of Graph Attention Network
        """
        super(ST_GAT, self).__init__()

        print("Graph Attention")

        self.n_pred = out_channels
        self.heads = heads
        self.dropout = dropout
        self.n_nodes = n_nodes

        lstm1_hidden_size = 32
        lstm2_hidden_size = 128

        # single graph attentional layer with 8 attention heads
        self.gat = GATConv(in_channels=in_channels, out_channels=in_channels,
            heads=heads, dropout=0, concat=False)

        # add two LSTM layers
        self.lstm1 = torch.nn.LSTM(input_size=self.n_nodes, hidden_size=lstm1_hidden_size, num_layers=1)
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
        self.lstm2 = torch.nn.LSTM(input_size=lstm1_hidden_size, hidden_size=lstm2_hidden_size, num_layers=1)
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)

        # fully-connected neural network
        self.linear1 = torch.nn.Linear(lstm2_hidden_size, (self.n_nodes*self.n_pred)//2)
        self.linear2 = torch.nn.Linear((self.n_nodes*self.n_pred)//2, (self.n_nodes*self.n_pred))
        # self.linear3 = torch.nn.Linear( (self.n_nodes*self.n_pred)//2, self.n_nodes*self.n_pred)
        
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        # torch.nn.init.xavier_uniform_(self.linear3.weight)

    # def forward(self, data, device = "cuda"):

    #     # print(type(data))
    #     # print(data)
    #     # print(data.x.shape)
    #     # print(data.edge_index.shape)

    #     """
    #     Forward pass of the ST-GAT model
    #     :param data Data to make a pass on
    #     :param device Device to operate on
    #     """
    #     x, edge_index = data.x, data.edge_index
    #     # apply dropout
    #     if device == 'cuda':
    #         x = torch.FloatTensor(x)
    #     else:
    #         x = torch.cuda.FloatTensor(x)

    #     # gat layer: output of gat: [11400, 12]
    #     x = self.gat(x, edge_index)
    #     x = F.dropout(x, self.dropout, training=self.training)

    #     # RNN: 2 LSTM
    #     # [batchsize*n_nodes, seq_length] -> [batch_size, n_nodes, seq_length]
    #     batch_size = data.num_graphs
    #     n_node = int(data.num_nodes/batch_size)
    #     x = torch.reshape(x, (batch_size, n_node, data.num_features))
    #     # for lstm: x should be (seq_length, batch_size, n_nodes)
    #     # sequence length = 12, batch_size = 50, n_node = 228
    #     x = torch.movedim(x, 2, 0)
    #     # [12, 50, 228] -> [12, 50, 32]
    #     x, _ = self.lstm1(x)
    #     # [12, 50, 32] -> [12, 50, 128]
    #     x, _ = self.lstm2(x)

    #     # Output contains h_t for each timestep, only the last one has all input's accounted for
    #     # [12, 50, 128] -> [50, 128]
    #     x = torch.squeeze(x[-1, :, :])
    #     # [50, 128] -> [50, 228*9]
    #     x = self.linear(x)

    #     # Now reshape into final output
    #     s = x.shape
    #     # [50, 228*9] -> [50, 228, 9]
    #     x = torch.reshape(x, (s[0], self.n_nodes, self.n_pred))
    #     # [50, 228, 9] ->  [11400, 9]
    #     x = torch.reshape(x, (s[0]*self.n_nodes, self.n_pred))
    #     return x


    def forward(self, data, device="cuda"):
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        batch_size = data.num_graphs

        # Apply GAT and dropout
        x = self.gat(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)

        # Correct reshaping for LSTM input and processing
        x = x.view(batch_size, self.n_nodes, -1).permute(2, 0, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Take the last timestep's output and feed it to a linear layer
        x = x[-1]
        x = self.linear1(x)
        x = self.linear2(x)
        # x = self.linear3(x)

        x = x.view(batch_size, self.n_nodes, self.n_pred)

        return x


# Some other Layers like GCN, GraphSAGE, GIN ... (for modelling) -->

class ST_GCN(torch.nn.Module):

    def __init__(self, in_channels, out_channels, n_nodes, dropout=0.0):

        super(ST_GCN, self).__init__()

        print("Graph Convolution Network")

        self.n_pred = out_channels
        self.dropout = dropout
        self.n_nodes = n_nodes

        self.n_preds = 15
        lstm1_hidden_size = 32
        lstm2_hidden_size = 128

        # GraphSAGE layer
        self.graph_conv = GCNConv(in_channels, in_channels)

        # add two LSTM layers
        self.lstm1 = torch.nn.LSTM(input_size=self.n_nodes, hidden_size=lstm1_hidden_size, num_layers=1)
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
        self.lstm2 = torch.nn.LSTM(input_size=lstm1_hidden_size, hidden_size=lstm2_hidden_size, num_layers=1)
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)

        # fully-connected neural network
        self.linear = torch.nn.Linear(lstm2_hidden_size, self.n_nodes*self.n_pred)
        torch.nn.init.xavier_uniform_(self.linear.weight)


    def forward(self, data, device="cuda"):

            # print(f" Shape of X in data: {data.x.shape}")

            # print(f" Shape of edge_index in data: {data.edge_index.shape}")


            x, edge_index = data.x.to(device), data.edge_index.to(device)
            batch_size = data.num_graphs

            # Apply GraphSAGE and dropout
            x = self.graph_conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

            # print(f" Shape after applying GAT layer (before changing shape): {x.shape}")


            # Correct reshaping for LSTM input and processing
            x = x.view(batch_size, self.n_nodes, -1).permute(2, 0, 1)

            # print(f" Shape before passing it into LSTM layer 1: {x.shape}")

            x, _ = self.lstm1(x)

            # print(f" Shape before passing it into LSTM layer 2: {x.shape}")

            x, _ = self.lstm2(x)

            # Take the last timestep's output and feed it to a linear layer
            x = x[-1]
            x = self.linear(x)
            x = x.view(batch_size, self.n_nodes, self.n_preds)

            return x


# GIN -->

class ST_GIN(torch.nn.Module):

    def __init__(self, in_channels, out_channels, n_nodes, dropout=0.0):

        super(ST_GIN, self).__init__()

        print("Graph Isomorphic Network")

        self.n_pred = out_channels
        self.dropout = dropout
        self.n_nodes = n_nodes

        self.n_preds = 15
        lstm1_hidden_size = 32
        lstm2_hidden_size = 128

        # Define an MLP for GIN
        mlp = MLP([in_channels, in_channels, in_channels])

        # GIN layer
        self.graph_conv = GINConv(mlp)

        # add two LSTM layers
        self.lstm1 = torch.nn.LSTM(input_size=self.n_nodes, hidden_size=lstm1_hidden_size, num_layers=1)
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
        self.lstm2 = torch.nn.LSTM(input_size=lstm1_hidden_size, hidden_size=lstm2_hidden_size, num_layers=1)
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)

        # fully-connected neural network
        self.linear = torch.nn.Linear(lstm2_hidden_size, self.n_nodes*self.n_pred)
        torch.nn.init.xavier_uniform_(self.linear.weight)


    def forward(self, data, device="cuda"):

            # print(f" Shape of X in data: {data.x.shape}")

            # print(f" Shape of edge_index in data: {data.edge_index.shape}")

            x, edge_index = data.x.to(device), data.edge_index.to(device)
            batch_size = data.num_graphs

            # Apply GraphSAGE and dropout
            x = self.graph_conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

            # print(f" Shape after applying GAT layer (before changing shape): {x.shape}")


            # Correct reshaping for LSTM input and processing
            x = x.view(batch_size, self.n_nodes, -1).permute(2, 0, 1)

            # print(f" Shape before passing it into LSTM layer 1: {x.shape}")

            x, _ = self.lstm1(x)

            # print(f" Shape before passing it into LSTM layer 2: {x.shape}")

            x, _ = self.lstm2(x)

            # Take the last timestep's output and feed it to a linear layer
            x = x[-1]
            x = self.linear(x)
            x = x.view(batch_size, self.n_nodes, self.n_preds)

            return x


# GraphSAGE -->

class ST_GraphSAGE(torch.nn.Module):

    def __init__(self, in_channels, out_channels, n_nodes, dropout=0.0):

        super(ST_GraphSAGE, self).__init__()

        print("GraphSAGE")

        self.n_pred = out_channels
        self.dropout = dropout
        self.n_nodes = n_nodes

        self.n_preds = 15
        lstm1_hidden_size = 32
        lstm2_hidden_size = 128

        # GraphSAGE layer
        self.graph_conv = SAGEConv(in_channels, in_channels, normalize=True, root_weight=True)

        # add two LSTM layers
        self.lstm1 = torch.nn.LSTM(input_size=self.n_nodes, hidden_size=lstm1_hidden_size, num_layers=1)
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
        self.lstm2 = torch.nn.LSTM(input_size=lstm1_hidden_size, hidden_size=lstm2_hidden_size, num_layers=1)
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)

        # fully-connected neural network
        self.linear = torch.nn.Linear(lstm2_hidden_size, self.n_nodes*self.n_pred)
        torch.nn.init.xavier_uniform_(self.linear.weight)


    def forward(self, data, device="cuda"):

            # print(f" Shape of X in data: {data.x.shape}")

            # print(f" Shape of edge_index in data: {data.edge_index.shape}")


            x, edge_index = data.x.to(device), data.edge_index.to(device)
            batch_size = data.num_graphs

            # Apply GraphSAGE and dropout
            x = self.graph_conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

            # print(f" Shape after applying GAT layer (before changing shape): {x.shape}")


            # Correct reshaping for LSTM input and processing
            x = x.view(batch_size, self.n_nodes, -1).permute(2, 0, 1)

            # print(f" Shape before passing it into LSTM layer 1: {x.shape}")

            x, _ = self.lstm1(x)

            # print(f" Shape before passing it into LSTM layer 2: {x.shape}")

            x, _ = self.lstm2(x)

            # Take the last timestep's output and feed it to a linear layer
            x = x[-1]
            x = self.linear(x)
            x = x.view(batch_size, self.n_nodes, self.n_preds)

            return x
    


# A3TGCN -->

class ST_A3TGCN(torch.nn.Module):

    def __init__(self, in_channels, out_channels, n_nodes, dropout=0.0):

        super(ST_A3TGCN, self).__init__()

        print("A3TGCN")

        self.n_pred = out_channels
        self.dropout = dropout
        self.n_nodes = n_nodes

        self.n_preds = 15
        lstm1_hidden_size = 32
        lstm2_hidden_size = 128

        # A3TGCN layer
        self.graph_conv = A3TGCN(in_channels, in_channels, periods=96)

        # add two LSTM layers
        self.lstm1 = torch.nn.LSTM(input_size=self.n_nodes, hidden_size=lstm1_hidden_size, num_layers=1)
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
        self.lstm2 = torch.nn.LSTM(input_size=lstm1_hidden_size, hidden_size=lstm2_hidden_size, num_layers=1)
        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)

        # fully-connected neural network
        self.linear = torch.nn.Linear(lstm2_hidden_size, self.n_nodes*self.n_pred)
        torch.nn.init.xavier_uniform_(self.linear.weight)


    def forward(self, data, device="cuda"):

            # print(f" Shape of X in data: {data.x.shape}")

            # print(f" Shape of edge_index in data: {data.edge_index.shape}")


            x, edge_index = data.x.to(device), data.edge_index.to(device)
            batch_size = data.num_graphs

            # Apply A3TGCN and dropout
            x = self.graph_conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

            # print(f" Shape after applying GAT layer (before changing shape): {x.shape}")


            # Correct reshaping for LSTM input and processing
            x = x.view(batch_size, self.n_nodes, -1).permute(2, 0, 1)

            # print(f" Shape before passing it into LSTM layer 1: {x.shape}")

            x, _ = self.lstm1(x)

            # print(f" Shape before passing it into LSTM layer 2: {x.shape}")

            x, _ = self.lstm2(x)

            # Take the last timestep's output and feed it to a linear layer
            x = x[-1]
            x = self.linear(x)
            x = x.view(batch_size, self.n_nodes, self.n_preds)

            return x
    

# 1st try
    
# class ST_GAT_Enhanced(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, n_nodes, heads=8, num_gat_layers=2, num_transformer_layers=2, dropout=0.1):
#         super(ST_GAT_Enhanced, self).__init__()

#         print("ST_GAT_Enhanced Model")

#         self.n_pred = out_channels
#         self.heads = heads
#         self.dropout = dropout
#         self.n_nodes = n_nodes

#         self.n_preds = 15
#         transformer_hidden_size = 64

#         # Multiple graph attention layers
#         self.gat_layers = torch.nn.ModuleList([GATConv(in_channels if i == 0 else in_channels, in_channels, heads=heads, dropout=dropout, concat=False) for i in range(num_gat_layers)])

#         # Transformer encoder layers
#         d_model = 60  # Must be divisible by nhead
#         nhead = 4     # Adjust nhead to be divisible by d_model
#         encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_transformer_layers)

#         # Fully-connected neural network
#         self.linear = torch.nn.Linear(d_model, self.n_nodes * self.n_pred)
#         torch.nn.init.xavier_uniform_(self.linear.weight)

#     def forward(self, data, device="cuda"):
#         x, edge_index = data.x.to(device), data.edge_index.to(device)
#         batch_size = data.num_graphs

#         # Apply multiple GAT layers with residual connections
#         for gat_layer in self.gat_layers:
#             x = gat_layer(x, edge_index) + x
#             x = F.dropout(x, self.dropout, training=self.training)

#         # Reshape for Transformer input and apply Transformer encoder
#         x = x.view(batch_size, self.n_nodes, -1).permute(1, 0, 2)
#         x = self.transformer_encoder(x)

#         # Take the last timestep's output and feed it to a linear layer
#         x = x[-1]
#         x = self.linear(x)
#         x = x.view(batch_size, self.n_nodes, self.n_preds)

#         return x


# 2nd try

# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GATConv
# from torch.nn import TransformerEncoder, TransformerEncoderLayer

# class ST_GAT_Enhanced(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, n_nodes, heads=8, num_gat_layers=4, num_transformer_layers=2, dropout=0.1):
#         super(ST_GAT_Enhanced, self).__init__()

#         print("ST_GAT_Enhanced Model")

#         self.n_pred = out_channels
#         self.heads = heads
#         self.dropout = dropout
#         self.n_nodes = n_nodes

#         self.n_preds = 15
#         self.transformer_hidden_size = 64

#         # Define the (GAT, Linear) blocks with skip connections
#         self.gat_linear_blocks = torch.nn.ModuleList()
#         for i in range(num_gat_layers):
#             gat_in = in_channels if i == 0 else in_channels
#             gat_out = in_channels
#             linear_in = in_channels
#             linear_out = in_channels

#             self.gat_linear_blocks.append(GATConv(gat_in, gat_out, heads=heads, dropout=dropout, concat=False))
#             self.gat_linear_blocks.append(torch.nn.Linear(linear_in, linear_out))

#         # Transformer encoder layers
#         d_model = in_channels  # Set d_model to in_channels for compatibility
#         nhead = 4              # Adjust nhead to ensure divisibility by d_model
#         encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_transformer_layers)

#         # Fully-connected neural network
#         self.linear = torch.nn.Linear(d_model, self.n_nodes * self.n_pred)
#         torch.nn.init.xavier_uniform_(self.linear.weight)

#     def forward(self, data, device="cuda"):
#         x, edge_index = data.x.to(device), data.edge_index.to(device)
#         batch_size = data.num_graphs

#         # Apply (GAT, Linear) blocks with residual connections
#         residual = x
#         for i in range(0, len(self.gat_linear_blocks), 2):
#             x = self.gat_linear_blocks[i](x, edge_index)
#             x = F.dropout(x, self.dropout, training=self.training)
#             x = self.gat_linear_blocks[i + 1](x)
#             if i in [2, 6]:  # Adding residual connections at the second and fourth blocks
#                 x += residual
#                 residual = x

#         # Reshape for Transformer input and apply Transformer encoder
#         x = x.view(batch_size, self.n_nodes, -1).permute(1, 0, 2)
#         x = self.transformer_encoder(x)

#         # Take the last timestep's output and feed it to a linear layer
#         x = x[-1]
#         x = self.linear(x)
#         x = x.view(batch_size, self.n_nodes, self.n_preds)

#         return x


# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GATConv
# from torch.nn import TransformerEncoder, TransformerEncoderLayer

# class ST_GAT_Enhanced(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, n_nodes, heads=8, num_gat_layers=4, num_transformer_layers=4, dropout=0.1):
#         super(ST_GAT_Enhanced, self).__init__()

#         print("ST_GAT_Enhanced Model")

#         self.n_pred = out_channels
#         self.heads = heads
#         self.dropout = dropout
#         self.n_nodes = n_nodes

#         self.n_preds = 15
#         self.transformer_hidden_size = 64
#         higher_dim = 128  # Define a higher dimension for the hidden layers

#         # Define the (Linear1, GAT, Linear2) blocks with skip connections
#         self.gat_linear_blocks = torch.nn.ModuleList()
#         for i in range(num_gat_layers):
#             self.gat_linear_blocks.append(torch.nn.Linear(in_channels if i == 0 else in_channels, higher_dim))
#             self.gat_linear_blocks.append(GATConv(higher_dim, higher_dim, heads=heads, dropout=dropout, concat=False))
#             self.gat_linear_blocks.append(torch.nn.Linear(higher_dim, in_channels))

#         # Transformer encoder layers
#         d_model = in_channels  # Set d_model to in_channels for compatibility
#         nhead = 4              # Adjust nhead to ensure divisibility by d_model
#         encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_transformer_layers)

#         # Fully-connected neural network
#         self.linear = torch.nn.Linear(d_model, self.n_nodes * self.n_pred)
#         torch.nn.init.xavier_uniform_(self.linear.weight)

#     def forward(self, data, device="cuda"):
#         num_transformer_layers=4
#         x, edge_index = data.x.to(device), data.edge_index.to(device)
#         batch_size = data.num_graphs

#         # Apply (Linear1, GAT, Linear2) blocks with residual connections
#         residual = x
#         for i in range(0, len(self.gat_linear_blocks), 3):
#             x = self.gat_linear_blocks[i](x)  # Linear1
#             x = F.relu(x)
#             x = self.gat_linear_blocks[i + 1](x, edge_index)  # GAT
#             x = F.dropout(x, self.dropout, training=self.training)
#             x = self.gat_linear_blocks[i + 2](x)  # Linear2
#             if i in [3, 9]:  # Adding residual connections at the second and fourth blocks
#                 x += residual
#                 residual = x

#         # Reshape for Transformer input and apply Transformer encoder
#         x = x.view(batch_size, self.n_nodes, -1).permute(1, 0, 2)
#         residual = x
#         for i in range(num_transformer_layers):
#             x = self.transformer_encoder.layers[i](x)
#             if i in [1, 3]:  # Adding residual connections at the second and fourth transformer layers
#                 x += residual
#                 residual = x

#         # Take the last timestep's output and feed it to a linear layer
#         x = x[-1]
#         x = self.linear(x)
#         x = x.view(batch_size, self.n_nodes, self.n_preds)

#         return x
