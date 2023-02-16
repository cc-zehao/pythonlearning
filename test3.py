from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
print(len(dataset))
print(dataset.num_classes)
print(dataset.num_features)