import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool, NNConv



class InvoiceGCN(torch.nn.Module):
    def __init__(self, in_feats, edge_feats, hidden_feats, out_feats):
        super(InvoiceGCN, self).__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(edge_feats, in_feats * hidden_feats))
        self.conv1 = NNConv(in_feats, hidden_feats, nn1, aggr='mean')

        nn2 = torch.nn.Sequential(torch.nn.Linear(edge_feats, hidden_feats * hidden_feats))
        self.conv2 = NNConv(hidden_feats, hidden_feats, nn2, aggr='mean')

        nn3 = torch.nn.Sequential(torch.nn.Linear(edge_feats, hidden_feats * hidden_feats))
        self.conv3 = NNConv(hidden_feats, hidden_feats, nn3, aggr='mean')

        nn4 = torch.nn.Sequential(torch.nn.Linear(edge_feats, hidden_feats * out_feats))
        self.conv4 = NNConv(hidden_feats, out_feats, nn4, aggr='mean')

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x, edge_index, edge_attr):  # Add edge_attr argument
        x = self.conv1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index, edge_attr)
        return x

