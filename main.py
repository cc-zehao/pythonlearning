# import torch
# from torch import Tensor
# from torch_geometric.nn import GCNConv
# from torch_geometric.datasets import Planetoid
#
# dataset = Planetoid(root='./data/Cora', name='Cora')
#
# class GCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)
#
#     def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
#         # x: Node feature matrix of shape [num_nodes, in_channels]
#         # edge_index: Graph connectivity matrix of shape [2, num_edges]
#         x = self.conv1(x, edge_index).relu()
#         x = self.conv2(x, edge_index)
#         return x
#
# data=dataset[0]
#
# model = GCN(dataset.num_features, 16, dataset.num_classes)
# print(model)
# print(data)
# print(data.x)
# print(data.edge_index)

# -----------------------------------------------------------------------


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot  as plt

def visualize_graph(G,color):
    plt.figure(figsize=(65,65))
    plt.xticks([])
    plt.yticks([])
    camp1=plt.get_cmap('Set2')
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False, nod_color=color, cmap=camp1)
    plt.show()

class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 32)
        self.conv2 = GCNConv(32, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.conv1(x, edge_index)
        # 在GCNConv（）中包含了将邻接矩阵加上自联结并且计算度矩阵的过程，不需要手动加入
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


dataset = Planetoid(root='./data/Cora', name='Cora')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# G=to_networkx(dataset[0],to_undirected=True)
# visualize_graph(G, color=dataset[0].y)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


model.eval()
pred = model(data).argmax(dim=1)
print(f'pred:{pred}:')
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')


