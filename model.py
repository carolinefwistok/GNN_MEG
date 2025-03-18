import torch
from torch.nn import Linear, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv


class GNN(torch.nn.Module):
    '''
    Defines the graph neural network (GNN) model that is trained and tested by train.py
    '''

    def __init__(self, n_layers, dropout_rate, conv1_hidden_channels, conv2_hidden_channels, conv3_hidden_channels, conv4_hidden_channels, dataset, top_k=None, threshold=None):
        '''
        Initializes the layers of the GNN.

        INPUTS:
            - n_layers                  : Number of layers in the GNN
            - dropout_rate              : Dropout rate for the GNN
            - conv1_hidden_channels     : Integer defining the number of hidden channels for the first convolutional layer
            - conv2_hidden_channels     : Integer defining the number of hidden channels for the second convolutional layer
            - conv3_hidden_channels     : Integer defining the number of hidden channels for the third convolutional layer
            - conv4_hidden_channels     : Integer defining the number of hidden channels for the fourth convolutional layer
            - dataset                   : Dataset of graphs
            - top_k                     : Optional parameter for top_k
            - threshold                 : Optional parameter for threshold
        OUTPUT: N/A
        '''

        torch.manual_seed(1234567)

        # Retrieve the basic functionality from torch.nn.Module
        super(GNN, self).__init__()

        # Define dropout rate
        self.ds = dropout_rate

        # Define the convolutional layers and batch normalization layers
        self.n_layers = n_layers
        self.bn1 = BatchNorm1d(conv1_hidden_channels)
        self.bn2 = BatchNorm1d(conv2_hidden_channels)

        # Define convolutional layers
        self.conv1 = GCNConv(dataset.num_node_features, conv1_hidden_channels)
        self.conv2 = GCNConv(conv1_hidden_channels, conv2_hidden_channels)

        if n_layers >= 2:
            self.conv3 = GCNConv(conv2_hidden_channels, conv3_hidden_channels)
            self.bn3 = BatchNorm1d(conv3_hidden_channels)
        if n_layers >= 3:
            self.conv4 = GCNConv(conv3_hidden_channels, conv4_hidden_channels)
            self.bn4 = BatchNorm1d(conv4_hidden_channels)

        # Define the final linear layer to compromise features into 2 classes (ON or OFF)
        if n_layers == 2:
            self.lin = Linear(conv2_hidden_channels, dataset.num_classes)
        elif n_layers == 3:
            self.lin = Linear(conv3_hidden_channels, dataset.num_classes)
        elif n_layers == 4:
            self.lin = Linear(conv4_hidden_channels, dataset.num_classes)
        else:
            raise ValueError('The number of layers exceeds 4!')

    def forward(self, x, edge_index, edge_attr, batch):
        '''
        Performs the layers initialized above.
        
        INPUTS:
            - x             : Torch tensor object of node feature matrix (PSD)
            - edge_index    : Torch tensor object of edges (indices of connected nodes) 
            - edge_attr     : Torch tensor object of edge features (computed with connectivity method)
            - batch         : Torch tensor object of batch indices
            - temperature   : Float defining the temperature for the softmax function
        
        OUTPUT:
            - x             : Torch tensor object of matrix containing the model's predictions for each graph
        '''

        # Node embeddings (message passing)
        x = self.conv1(x, edge_index, edge_attr)  # Apply convolutional layer
        x = self.bn1(x)  # Apply batch normalization
        x = F.relu(x)  # Apply ReLU activation function
        x = F.dropout(x, p=self.ds, training=self.training)  # Apply dropout

        if self.n_layers >= 2:
            x = self.conv2(x, edge_index, edge_attr)
            x = self.bn2(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.ds, training=self.training)

        if self.n_layers >= 3:
            x = self.conv3(x, edge_index, edge_attr)
            x = self.bn3(x)  # Apply batch normalization
            x = F.relu(x)
            x = F.dropout(x, p=self.ds, training=self.training)

        if self.n_layers == 4:
            x = self.conv4(x, edge_index, edge_attr)
            x = self.bn4(x)  # Apply batch normalization
            x = F.relu(x)

        # Readout layer
        x = global_mean_pool(x, batch)

        # Apply final classifier 
        x = F.dropout(x, p=self.ds, training=self.training)
        x = self.lin(x)
        
        # Apply softmax function to the output with temperature parameter
        # x = x / temperature

        return x
