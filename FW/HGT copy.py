# 创建一个 HeteroConv 模型作为编码器
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.conv import HeteroConv, HGTConv, RGCNConv, FastRGCNConv, GATConv, SAGEConv

class HeteroGraphConvModel(nn.Module):
    def __init__(self, model, in_channels, hidden_channels, out_channels, metadata, args):
        super(HeteroGraphConvModel, self).__init__()
        self.model = model
        
        self.des_size = args.des_num
        self.tweet_size = args.tweet_num
        self.num_prop_size = args.prop_num
        self.cat_prop_size = args.cat_num

        self.linear_relu_des = nn.Sequential(
            nn.Linear(args.des_size, int(args.embedding_dim / 4)),
            nn.LeakyReLU(),
            nn.Dropout(args.dropout))
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(args.tweet_size, int(args.embedding_dim / 4)),
            nn.LeakyReLU(),
            nn.Dropout(args.dropout))
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(args.num_prop_size, int(args.embedding_dim / 4)),
            nn.LeakyReLU(),
            nn.Dropout(args.dropout))
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(args.cat_prop_size, int(args.embedding_dim / 4)),
            nn.LeakyReLU(),
            nn.Dropout(args.dropout))
        self.linear = nn.ModuleDict({node_type: nn.Linear(in_channels[node_type], hidden_channels)
                                     if node_type != 'user' else nn.Linear(args.embedding_dim, hidden_channels)
                                     for node_type in in_channels})
        self.LReLU = nn.LeakyReLU()
        self.Hetero_Conv_list = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(args.num_layer - 1):
            self.Hetero_Conv_list.append(self.choice_basic_model(hidden_channels, hidden_channels, metadata))
            self.batch_norms.append(BatchNorm(hidden_channels, momentum=args.momentum))
        self.Hetero_Conv_list.append(self.choice_basic_model(hidden_channels, out_channels, metadata))
        self.batch_norms.append(BatchNorm(out_channels, momentum=args.momentum))
        self.dropout = nn.Dropout(p=args.dropout)

        self.init_weight()

    def forward(self, graph):
        num_prop = graph['user'].x[:, :self.num_prop_size]
        cat_prop = graph['user'].x[:, self.num_prop_size:(self.num_prop_size+self.cat_prop_size)]
        des = graph['user'].x[:, (self.num_prop_size+self.cat_prop_size):
                                 (self.num_prop_size+self.cat_prop_size+self.des_size)]
        tweet = graph['user'].x[:, (self.num_prop_size+self.cat_prop_size+self.des_size):]
        user_features_des = self.linear_relu_des(des)
        user_features_tweet = self.linear_relu_tweet(tweet)
        user_features_numeric = self.linear_relu_num_prop(num_prop)
        user_features_bool = self.linear_relu_cat_prop(cat_prop)
        graph['user'].x = torch.cat((user_features_numeric, user_features_bool,
                                     user_features_des, user_features_tweet), dim=1)
        node_feature = graph.x_dict
        edge_index = graph.edge_index_dict

        node_embedding = {node_type: self.dropout(self.LReLU(self.linear[node_type](feature)))
                          for node_type, feature in node_feature.items()}
        # 卷积、批量归一化
        for Conv, batch_norm in zip(self.Hetero_Conv_list, self.batch_norms):
            node_embedding = Conv(node_embedding, edge_index)
            # node_embedding = {node_type: batch_norm(embedding) for node_type, embedding in node_embedding.items()}

        node_embedding = {node_type: torch.cat((embedding, node_feature[node_type]), dim=1)
                          for node_type, embedding in node_embedding.items()}
        return node_embedding

    def choice_basic_model(self, in_channels, out_channels, metadata):
        if self.model == "HGT":
            return HGTConv(in_channels=in_channels, out_channels=out_channels, metadata=metadata, heads=2, group='sum')
        elif self.model == "GAT":
            return HeteroConv({edge_type: GATConv((-1, -1), out_channels, add_self_loops=False)
                               for edge_type in metadata[1]}, aggr='sum')
        elif self.model == "SAGE":
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

    def init_first_layer(self, weight):
        self.load_state_dict(weight, strict=False)