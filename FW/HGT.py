# 创建一个 HeteroConv 模型作为编码器
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.conv import HeteroConv, HGTConv, RGCNConv, FastRGCNConv, GATConv, SAGEConv

class HGT(nn.Module):
    def __init__(self, args, metadata):
        super(HGT, self).__init__()
        self.hetero_type = args.hetero_type   
        self.des_size = args.des_num
        self.tweet_size = args.tweet_num
        self.num_prop_size = args.prop_num
        self.cat_prop_size = args.cat_num
        self.dropout = args.dropout
        self.node_num = args.node_num
        self.pe = args.pe
        self.pf = args.pf
        input_dimension = args.input_dim            ## 128
        embedding_dimension = args.hidden_dim       ## 32
        
        self.linear_relu_des = nn.Sequential(
            nn.Linear(self.des_size, int(input_dimension / 4)),
            nn.LeakyReLU(),
            nn.Dropout(args.dropout))
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(self.tweet_size, int(input_dimension / 4)),
            nn.LeakyReLU(),
            nn.Dropout(args.dropout))
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(self.num_prop_size, int(input_dimension / 4)),
            nn.LeakyReLU(),
            nn.Dropout(args.dropout))
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(self.cat_prop_size, int(input_dimension / 4)),
            nn.LeakyReLU(),
            nn.Dropout(args.dropout))
        
        #! 这里进行了维度的缩放 128->32
        self.linear = nn.Linear(input_dimension, embedding_dimension)
        self.LReLU = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=args.dropout)
        
        self.Hetero_Conv_list = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        #! 创建模型
        self.hgt1 = HGTConv(in_channels=embedding_dimension, out_channels=embedding_dimension,
                            metadata=metadata, heads=2)
        self.hgt1.to('cuda:0')
        self.hgt2 = HGTConv(in_channels=embedding_dimension, out_channels=embedding_dimension, 
                            metadata=metadata, heads=2)
        self.hgt2.to('cuda:0')
        #! 权重初始化
        self.init_weight()

    def forward(self, data):
        x = data['user'].x
        num_prop = x[:, :self.num_prop_size]
        tweet = x[:, self.num_prop_size:self.num_prop_size + self.tweet_size]
        cat_prop = x[:,
                     self.num_prop_size + self.tweet_size:self.num_prop_size +
                     self.tweet_size + self.cat_prop_size]
        des = x[:, self.num_prop_size + self.tweet_size + self.cat_prop_size:]
        
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        
        #! x 的维度为 128
        x = torch.cat((d, t, n, c), dim=1)
            
        #! 获得 x, edge_index 的特征
        node_feature, edge_index = data.x_dict, data.edge_index_dict
        #! 先进行维度缩放
        node_feature['user']=self.LReLU(self.linear(x))
        #! 然后过两层 hgt
        node_embedding = self.hgt1(node_feature, edge_index)
        node_embedding['user'] = self.LReLU(self.dropout(node_embedding['user'])) 
        node_embedding = self.hgt2(node_embedding, edge_index)
        node_embedding['user'] = self.LReLU(self.dropout(node_embedding['user']))
        # #! 将经过 hgt 的 embedding 和经过 hgt 之前的 embedding 拼接在一起, 选择要不要保存呢?
        node_embedding = {node_type: torch.cat((embedding, node_feature[node_type]), dim=1)
                          for node_type, embedding in node_embedding.items()}
        #! 只返回 'user' 的 embedding
        return node_embedding['user']

    def choice_basic_model(self, in_channels, out_channels, metadata):
        if self.hetero_type == "HGT":
            return HGTConv(in_channels=in_channels, out_channels=out_channels, metadata=metadata, heads=2, group='sum')
        elif self.hetero_type == "GAT":
            return HeteroConv({edge_type: GATConv((-1, -1), out_channels, add_self_loops=False)
                               for edge_type in metadata[1]}, aggr='sum')
        elif self.hetero_type == "SAGE":
            return HeteroConv({edge_type: SAGEConv((-1, -1), out_channels) for edge_type in metadata[1]}, aggr='sum')

    def init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.fill_(0.0)
            elif isinstance(module, dict):
                for layer in module.values():
                    torch.nn.init.kaiming_uniform_(layer.weight.data)
                    if layer.bias is not None:
                        layer.bias.data.fill_(0.0)