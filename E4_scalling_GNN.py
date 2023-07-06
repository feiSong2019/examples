# partition graph into several subgraph---mini-batch
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset=Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())
data=dataset[0]

from torch_geometric.loader import ClusterData, ClusterLoader
torch.manual_seed(123456)
cluster_data=ClusterData(data, num_parts=128)                       ### partition
train_loader=ClusterLoader(cluster_data,batch_size=32,shuffle=True) ### batch


































































