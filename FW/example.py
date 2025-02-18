import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import HGTConv
from torch_geometric.data import HeteroData

# 假设有 10 个用户
num_users = 10
num_node_features = 16
num_classes = 3

# 构建一个异构图数据结构
data = HeteroData()

# 所有节点都是 'user' 类型的节点，随机初始化特征
data['user'].x = torch.randn((num_users, num_node_features))

# 添加两种不同的边关系：'follows' 和 'messages'
# follows 关系边
data['user', 'follows', 'user'].edge_index = torch.tensor([
    [0, 1, 2, 3, 4],
    [1, 2, 3, 4, 0]
], dtype=torch.long)

# messages 关系边
data['user', 'messages', 'user'].edge_index = torch.tensor([
    [0, 2, 3],
    [2, 3, 4]
], dtype=torch.long)

# 生成节点的标签
data['user'].y = torch.randint(0, num_classes, (num_users,))

# 定义 HGT 模型
class HGTModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, metadata):
        super(HGTModel, self).__init__()
        # 使用 HGTConv 图卷积
        self.conv1 = HGTConv(in_channels, hidden_channels, metadata, heads=num_heads, group='sum')
        self.conv2 = HGTConv(hidden_channels, out_channels, metadata, heads=num_heads, group='sum')
        self.lin = Linear(out_channels, num_classes)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        
        # 两层 HGTConv 进行特征传播
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)

        # 只对 'user' 类型的节点进行分类
        out = self.lin(x_dict['user'])
        return out

# 获取异构图的元数据
metadata = data.metadata()

# 初始化模型
model = HGTModel(in_channels=num_node_features, 
                 hidden_channels=32, 
                 out_channels=32, 
                 num_heads=2, 
                 metadata=metadata)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 模型训练
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data['user'].y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 模型评估
model.eval()
with torch.no_grad():
    out = model(data)
    pred = out.argmax(dim=1)
    correct = (pred == data['user'].y).sum()
    accuracy = int(correct) / num_users
    print(f'Accuracy: {accuracy:.4f}')
