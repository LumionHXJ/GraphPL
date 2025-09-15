import torch
from torch import nn, Tensor
from torch.nn import Parameter
from functools import reduce
import math
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor

def compute_class_dim(*channels):
    def lcm(a, b):
        return abs(a * b) // math.gcd(a, b)
    return reduce(lcm, channels)

def fetch_node_feature(cm_matrix, observed_mask):
    row, col = torch.nonzero(observed_mask, as_tuple=True)
    cm_feature = cm_matrix[row, col]
    return cm_feature

def get_impute_index(observed_mask, cid, mid):
    row, col = torch.nonzero(observed_mask, as_tuple=True)
    idx =  torch.nonzero((row == cid) & (col == mid), as_tuple=True)[0][0]
    return idx

def construct_edge(observed_mask):
    """
    client-modality matrix: (C, M), 1 for observed.
    return:
        client graph: sharing same clients (cross-modal)
        modal graph: sharing same modal (cross-client)
    """
    c, m = torch.nonzero(observed_mask, as_tuple=True)
    edge_client = (c[:, None] == c[None]).to(torch.long)
    edge_client = torch.stack(torch.nonzero(edge_client, as_tuple=True))
    edge_modal = (m[:, None] == m[None]).to(torch.long)
    edge_modal = torch.stack(torch.nonzero(edge_modal, as_tuple=True))
    return edge_client, edge_modal

def construct_graph(num_nodes, impute_idx, level=0):
    if level == 0: # fully connected
        return torch.nonzero(torch.ones((num_nodes, num_nodes))).transpose(0,1)
    elif level == 1: # undirect star graph
        edge_matrix = torch.eye(num_nodes)
        edge_matrix[impute_idx] = 1
        edge_matrix[:, impute_idx] = 1
        return torch.nonzero(edge_matrix).transpose(0,1)
    elif level == 2: # direct star graph
        edge_matrix = torch.eye(num_nodes)
        edge_matrix[:, impute_idx] = 1 
        return torch.nonzero(edge_matrix).transpose(0,1)


class CMFusion(nn.Module):
    def __init__(self, class_dim, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(class_dim, class_dim)
        self.m_fc = nn.Sequential(
             nn.Linear(class_dim, class_dim),
             nn.GELU()
        )
        self.o_fc = nn.Sequential(
            nn.Linear(class_dim, class_dim),
            nn.Dropout(dropout)
        )
    def forward(self, fc, fm):
        fc = self.c_fc(fc)
        fm = self.m_fc(fm)
        out = self.o_fc(fc * fm)
        return out

class GroupGCNConv(GCNConv):
    def __init__(self, channels, groups):
        super().__init__(channels, channels)
        self.weight = Parameter(torch.Tensor(groups, channels, channels))
        self.bias = Parameter(torch.Tensor(groups, channels))
        self.reset_parameters()
        self.groups = groups
        self.channels = channels
    
    def message(self, x_j: Tensor, edge_weight) -> Tensor:
        # x_j: N, G, E, d, edge_weight: E, G or E, 1
        if edge_weight is None:
            return x_j
        else:
            return edge_weight.permute(1,0).unsqueeze(-1)[None] * x_j
    
    def forward(self, x, edge_index, edge_weight: torch.Tensor = None) -> torch.Tensor:
        E = edge_index.shape[1]
        if edge_weight is None:
            edge_weight = torch.ones((E, 1), dtype=torch.float).to(x.device)
        if self.normalize:
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim),
                self.improved, add_self_loops=False, dtype=x.dtype) # self_loops already added
        N, V, D = x.shape
        x = x.view(N, V, self.groups, self.channels) # N, V, h, d
        x = torch.einsum('nvhd, hdx->nvhx', x, self.weight).permute(0,2,1,3).contiguous()

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        # input X: n, h, v, d
        # edge_weight: E, h
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None).permute(0,2,1,3)  # N, V, h, d

        if self.bias is not None:
            out += self.bias # N, V, h, d

        # ! shuffling 
        out = out.permute(0, 1, 3, 2).contiguous() # N, V, d, h

        return out.view(N, V, D)

class GCNBlock(nn.Module):
    def __init__(self, class_dim, dropout=0.1):
        super().__init__()
        self.gcn = GCNConv(class_dim, class_dim, cached=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge):
        x = self.gcn(x, edge)
        return self.dropout(self.relu(x))
    
class GATBlock(nn.Module):
    def __init__(self, class_dim, groups=4, dropout=0.1):
        super().__init__()
        self.gat = GATConv(class_dim * groups, class_dim, heads=groups, dropout=dropout)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge, weight=None):
        # x: N, V, D
        # edge: 2, E
        N, V, D = x.shape
        edge = edge.unsqueeze(-1).expand(-1, -1, N).clone()
        edge += torch.arange(0, N)[None, None].to(edge.device) * V
        edge = edge.reshape(2, -1)
        x = x.reshape(-1, D)
        x = self.gat(x, edge)
        x = self.dropout(self.act(x))
        return x.reshape(N, V, D)

class GroupGCNBlock(nn.Module):
    def __init__(self, class_dim, groups, dropout=0.1):
        super().__init__()
        self.gcn = GroupGCNConv(class_dim, groups)
        self.relu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge, weight=None):
        x = self.gcn(x, edge, weight)
        return self.dropout(self.relu(x))

class CMBlock(nn.Module):
    def __init__(self, gnn_dim, groups=None, dropout=0.1):
        super().__init__()
        self.client_gcn = GCNBlock(gnn_dim, dropout=dropout)
        self.modal_gcn = GCNBlock(gnn_dim, dropout=dropout)
        self.fusion = CMFusion(gnn_dim)
    def forward(self, x, edge_client, edge_modal):
        c = self.client_gcn(x, edge_client)
        m = self.modal_gcn(x, edge_modal)
        out = self.fusion(c, m) + x
        return out
    
class GroupCMBlock(nn.Module):
    def __init__(self, gnn_dim, groups=None, dropout=0.1):
        super().__init__()
        self.client_gcn = GroupGCNBlock(gnn_dim, groups, dropout=dropout)
        self.modal_gcn = GroupGCNBlock(gnn_dim, groups, dropout=dropout)
        self.fusion = CMFusion(gnn_dim)
        self.groups = groups
    def forward(self, x, edge_client, edge_modal):
        N, V, D = x.shape
        c = self.client_gcn(x, edge_client).view(N, V, self.groups, -1)
        m = self.modal_gcn(x, edge_modal).view(N, V, self.groups, -1)
        out = self.fusion(c, m).view(N, V, D) + x
        return out
    
class FFN(nn.Module):
    def __init__(self, class_dim, dropout=0.1):
        super().__init__()
        self.m_fc = nn.Sequential(
            nn.LayerNorm(class_dim),
            nn.Linear(class_dim, class_dim * 2),             
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(class_dim * 2, class_dim),
        )
    def forward(self, fm):
        fm = self.m_fc(fm)
        return fm

class GroupBlock(nn.Module):
    def __init__(self, gnn_dim, groups=1, dropout=0.1):
        super().__init__()
        self.modal_gcn = GroupGCNBlock(gnn_dim, groups, dropout=dropout)
        self.ffn = FFN(gnn_dim)
        self.norm = nn.LayerNorm(gnn_dim * groups)
        self.groups = groups
    def forward(self, x, edge_modal, edge_weight=None):
        N, V, D = x.shape
        m =  self.modal_gcn(x, edge_modal, edge_weight) + x
        out = self.ffn(m.view(N, V, self.groups, -1)).view(N, V, D) + m
        out = self.norm(out)
        return out