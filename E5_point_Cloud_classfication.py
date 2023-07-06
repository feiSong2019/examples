'''
1. visual mesh & point cloud
2. point cloud classfication with GNN
    1) handing dataset
    2) generate point cloud via mesh
    3) re-implement PointNet++: point cloud classfication/segmentation via GNN
        3.1 grouping phase
        3.2 neighborhood aggragation phase
        3.3 downsampling phase
3. Network Archirecture in min-batch fashion
4. Training Procedure
========others=======
5. Rotation-invariant PointNet Layer
6. Downsampling phase via farthest point sampling
'''
#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_mesh(pos, face):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], triangles=data.face.t(), antialiased=False)
    plt.show()


def visualize_points(pos, edge_index=None, index=None):
    fig = plt.figure(figsize=(4, 4))
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():
             src = pos[src].tolist()
             dst = pos[dst].tolist()
             plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black')
    if index is None:
        plt.scatter(pos[:, 0], pos[:, 1], s=50, zorder=1000)
    else:
       mask = torch.zeros(pos.size(0), dtype=torch.bool)
       mask[index] = True
       plt.scatter(pos[~mask, 0], pos[~mask, 1], s=50, color='lightgray', zorder=1000)
       plt.scatter(pos[mask, 0], pos[mask, 1], s=50, zorder=1000)
    plt.axis('off')
    plt.show()

from torch_geometric.datasets import GeometricShapes

dataset = GeometricShapes(root='data/GeometricShapes')
data=dataset[0]
visualize_mesh(data.pos, data.face)

#%%
import torch
from torch_geometric.transforms import SamplePoints

torch.manual_seed(42)
dataset.transform = SamplePoints(num=256)
data=dataset[0]
visualize_points(data.pos, data.edge_index)


# %%
# pointNet++
# s1: group nearby points via (knn or ball queries)
# s2: neighborhood aggragation phase: GNN layer
# s3: downsampling : pool

#phase1: grouping via dynamic graph generation

from torch_cluster import knn_graph

data=dataset[0]
data.edge_index=knn_graph(data.pos, k=6)
print(data.edge_index)
visualize_points(data.pos, edge_index=data.edge_index)



# %%
# phase2: neighbor aggregation
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(PointNetLayer, self).__init__(aggr='max')
        self.mlp=Sequential(Linear(in_channels+3, out_channels),
        ReLU(),
        Linear(out_channels, out_channels)
        )

    def forward(self, h, pos, edge_index):
        return self.propagate(edge_index, h=h, pos=pos)
    
    def message(self, h_j, pos_j, pos_i):
        input= pos_j - pos_i
        if h_j is not None:
            input=torch.cat([h_j, input], dim=-1)
        return self.mlp(input)

# network framework
import torch
import torch.nn.functional as F 
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool 

class PointNet(torch.nn.Module):
    def __init__(self):
        super(PointNet,self).__init__()
        
        torch.manual_seed(123456)
        self.conv1=PointNetLayer(3, 32)
        self.conv2=PointNetLayer(32,32)
        self.classifier=Linear(32, dataset.num_classes)

    def forward(self, pos, batch):
        edge_index=knn_graph(pos, k=16, batch=batch, loop=True)

        h=self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h=h.relu()
        h=self.conv2(h=h, pos=pos, edge_index=edge_index)
        h=h.relu()

        h=global_max_pool(h, batch) ###池化操作要观察batch的大小

        return self.classifier(h)
model= PointNet()
print(model)
# %%

## train
from torch_geometric.loader import DataLoader

train_dataset=GeometricShapes(root='data/GeometricShapes', train=True, transform=SamplePoints(128))
test_dataset=GeometricShapes(root='data/GeometricShapes', train=False, transform=SamplePoints(128))

train_loader=DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=10)

model=PointNet()
optimizer=torch.optim.Adam(model.parameters(), lr=0.01)
criterion=torch.nn.CrossEntropyLoss()

def train(model, optimizer, loader):
    model.train()
    total_loss=0

    for data in loader:
        optimizer.zero_grad()
        logits=model(data.pos, data.batch)
        loss=criterion(logits, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*data.num_graphs

    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(model, loader):
    model.eval()

    total_correct=0
    for data in loader:
        logits=model(data.pos, data.batch)
        pred=logits.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)

for epoch in range(1,51):
    loss = train(model, optimizer, train_loader)
    test_acc = test(model, test_loader)
    print(f'Epoch:{epoch:02d}, Loss:{loss:.4f}, Test Accuracy:{test_acc:.4f}')

# %%
