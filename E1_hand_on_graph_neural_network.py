'''=======goals========
1. init graph from data
2. visual graph
3. visual embedding
4. implement graph neural network



'''
#%%
#s1: graph initlize
from torch_geometric.datasets import KarateClub
dataset=KarateClub() # init dataset
print(f'Dataset: {dataset}')
print(f'Number of graphs:{len(dataset)}')
print(f'Number of features:{dataset.num_features}')
print(f'Number of classes:{dataset.num_classes}')
data=dataset[0] # Data



#s2:采用 to_networkx 直接对图进行可视化
#%matplotlib inline
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
def visualize_graph (G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,node_color=color, cmap="Set2")
    plt.show()
G=to_networkx(data, to_undirected=True)
visualize_graph(G, color=data.y)

def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h=h.detach().cpu().numpy()
    plt.scatter(h[:,0], h[:,1], s=140, c=color, cmap='Set2')
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()
# %%
#S3: implement graph neural networks
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv #GCNConv 实现具有归一化的 LXW 功能; 对node进行卷积，node的个数并不改变, 故edge_index并不改变

class GCN(torch.nn.Module):
    def __init__(self,dataset):
        super(GCN,self).__init__()
        torch.manual_seed(1234)
        self.conv1=GCNConv(dataset.num_features,4)
        self.conv2=GCNConv(4,4)
        self.conv3=GCNConv(4,2)
        self.classifier=Linear(2, dataset.num_classes)
    def forward(self,x,edge_index):
        h=self.conv1(x,edge_index)
        h=h.tanh()
        h=self.conv2(h,edge_index)
        h=h.tanh()
        h=self.conv3(h,edge_index)
        h=h.tanh()                  # Final GNN embedding space
        out=self.classifier(h)      # Apply a final linear classifier
        return out, h
model=GCN(dataset)
'''
print(model)
_,h=model(data.x, data.edge_index)
print(f'Embedding shape: {list(h.shape)}')
visualize_embedding(h, color=data.y)
'''

#S4: train model
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

def train(data):
    optimizer.zero_grad()
    out,h=model(data.x, data.edge_index)
    loss=criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss,h

import time
for epoch in range(401):
    loss, h= train(data)
    if epoch % 10 == 0:
        visualize_embedding(h, color=data.y, epoch=epoch, loss=loss)
        time.sleep(0.3)
# %%
