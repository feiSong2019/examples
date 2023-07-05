# node classfication (transductive learning)
#%%
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset=Planetoid(root='/home/songf/Desktop/songf/LearningGNNBook/', name='Cora', transform=NormalizeFeatures()) # 在下载时直接对数据集进行预处理
print()
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
data=dataset[0]
print(data)



# %%
# 仅依据节点内容预测节点分类:设计一个两层线性网络
import torch
from torch.nn import Linear
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, dataset, hidden_channels):
        super(MLP, self).__init__()
        torch.manual_seed(123456)
        self.lin1=Linear(dataset.num_features, hidden_channels)
        self.lin2=Linear(hidden_channels, dataset.num_classes)

    def forward(self, x):
        x=self.lin1(x)
        x=x.relu()
        x=F.dropout(x,p=0.5, training=self.training)
        x=self.lin2(x)
        return x
model=MLP(dataset, hidden_channels=16)
print(model)


criterion= torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out=model(data.x)
    loss=criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss
def test():
    model.eval()
    out=model(data.x)
    pred=out.argmax(dim=1)
    test_correct=pred[data.test_mask]==data.y[data.test_mask]
    test_acc=int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc

for epoch in range(1, 201):
    loss=train()
    test_acc=test()
    print(f'Epoch:{epoch:03d}, Loss:{loss:.4f}, test:{test_acc:.4f}')

# %%
## utilize GCN to implement model
from torch_geometric.nn import GCNConv
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(1234567)
        self.conv1=GCNConv(dataset.num_features,hidden_channels)
        self.conv2=GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x= self.conv1(x, edge_index)
        x=x.relu()
        x=F.dropout(x, p=0.5, training=self.training)
        x=self.conv2(x, edge_index)
        return x
model=GCN(hidden_channels=16)
#print(model)
optimizer= torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=5e-4)
criterion=torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out=model(data.x, data.edge_index)
    loss=criterion(out[data.train_mask],data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

def test():
    model.eval()
    out=model(data.x, data.edge_index)
    pred=out.argmax(dim=1)
    test_correct=pred[data.test_mask]==data.y[data.test_mask]
    test_acc=int(test_correct.sum())/int(data.test_mask.sum())
    return test_acc

for epoch in range(1,101):
    loss=train()
    test_acc=test()
    print(f'Epoch:{epoch:03d}, Loss: {loss:.4f}, test: {test_acc:.4f}')
# %%
