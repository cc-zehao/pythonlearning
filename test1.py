
import torch
from torch.nn import Linear
from torch import Tensor
from torch_geometric.nn import GCNConv
import numpy
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot  as plt

def visualize_graph(G,color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    camp1=plt.get_cmap('Set2')
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True, nod_color=color, cmap=camp1)
    plt.show()

def visualize_embedding(h,color,epoch=None,loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h=h.detach().cpu().numpy()
    plt.scatter(h[:,0],h[:,1],s=140,c=color,cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch:{epoch},Loss{loss.item():.4f}',fontsize=16)
    plt.show()



dataset = KarateClub()

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1=GCNConv(dataset.num_features,4)
        self.conv2=GCNConv(4,4)
        self.conv3=GCNConv(4,2)
        self.classifier=Linear(2,dataset.num_classes)

    def forward(self,x,edge_index):
        h=self.conv1(x,edge_index)
        h=h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        out =self.classifier(h)
        return  out,h


print(f'Dataset:{dataset}:')

data=dataset[0]
print(data)
edge_index=data.edge_index
# print(edge_index)
# print(edge_index.t())


print(data.y)
# G=to_networkx(data,to_undirected=True)
# visualize_graph(G, color=data.y)



model=GCN()
print(model)

# print(data.x,data.edge_index)
_,h=model(data.x,data.edge_index)
print(f'Embedding shape:{list(h.shape)}')

visualize_embedding(h,color=data.y)

import time
criterion =torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
# print(criterion)
# print(optimizer)
# print("模型参数:",(model.parameters()))
# for param in model.parameters():
#     print("参数类型：",type(param),"参数大小：",param.size())
def train(data):
    optimizer.zero_grad()
    out,h=model(data.x,data.edge_index)
    loss= criterion(out[data.train_mask],data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return  loss,h
for epoch in range(401):
    loss,h=train(data)
    if epoch % 50==0 :
        visualize_embedding(h,color=data.y,epoch=epoch,loss=loss)
        time.sleep(0.3)


