# Add self-loops to the adjacency matrix.
#
# Linearly transform node feature matrix.
#
# Compute normalization coefficients.
#
# Normalize node features in
#
# Sum up neighboring node features ("add" aggregation).
#
# Apply a final bias vector.
#
# E 节点数量 N
import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

edge_index = torch.tensor([[1,2,3],[0,0,0]],dtype=torch.long)
x=torch.tensor([[1],[1],[1],[1]],dtype=torch.float)
print(f'edge_index:{edge_index}:')
print(f'x:{x}:')

class GCNConv1(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        # self.lin = torch.nn.Linear(in_channels, out_channels)
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        print(f'add_self_loops,edge_index:{edge_index}:')
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
        print(f'linearly_transform:{x}:')

        # Step 3: Compute normalization.
        row, col = edge_index
        print(f'row:{row}:')
        print(f'col:{col}:')
        deg = degree(col, x.size(0), dtype=x.dtype)
        print(f'deg:{deg}:')
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        print(f'norm:{norm}:')

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)
        #
        # # Step 6: Apply a final bias vector.
        print(f'out:{out}:')
        print(f'bias:{self.bias}:')
        out += self.bias
        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        print(f'x_j:{x_j}:')
        print(f'norm.view:{norm.view(-1,1)}:')
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

conv=GCNConv1(1,2)
ret=conv(x,edge_index)
print(ret)