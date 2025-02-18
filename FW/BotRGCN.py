import torch
import torch.nn as nn
import torch.nn.functional as F

from BackBone.rgcn import RGCNConv

def edge_mask(edge_index, edge_attr, pe):
    # each edge has a probability of pe to be removed
    edge_index = edge_index.clone()
    edge_num = edge_index.shape[1]
    pre_index = torch.bernoulli(torch.ones(edge_num) * pe) == 0
    pre_index.to(edge_index.device)
    edge_index = edge_index[:, pre_index]
    edge_attr = edge_attr.clone()
    edge_attr = edge_attr[pre_index]
    return edge_index, edge_attr

class BotRGCN(nn.Module):

    def __init__(self, args):
        super(BotRGCN, self).__init__()
        self.des_size = args.des_num
        self.tweet_size = args.tweet_num
        self.num_prop_size = args.prop_num
        self.cat_prop_size = args.cat_num
        self.dropout = args.dropout
        self.node_num = args.node_num
        self.pe = args.pe
        self.pf = args.pf
        input_dimension = args.input_dim
        embedding_dimension = args.hidden_dim

        self.linear_relu_des = nn.Sequential(
            nn.Linear(self.des_size, int(input_dimension / 4)), nn.LeakyReLU())
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(self.tweet_size, int(input_dimension / 4)),
            nn.LeakyReLU())
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(self.num_prop_size, int(input_dimension / 4)),
            nn.LeakyReLU())
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(self.cat_prop_size, int(input_dimension / 4)),
            nn.LeakyReLU())

        self.linear_relu_input = nn.Sequential(
            nn.Linear(input_dimension, embedding_dimension),
            nn.PReLU(embedding_dimension))

        self.rgcn1 = RGCNConv(embedding_dimension,
                              embedding_dimension,
                              num_relations=args.num_relations)
        self.rgcn2 = RGCNConv(embedding_dimension,
                              embedding_dimension,
                              num_relations=args.num_relations)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension))

        self.relu = nn.LeakyReLU()

    def forward(self, data, return_attention=False):
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type

        if self.training:
            edge_index, edge_type = edge_mask(edge_index, edge_type, self.pe)

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
        x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)
        #! 如果 rgcn 是 FACNConv, 而不是 RGCNConv, 那么就用得上 return_attention 这个值
        # x = self.rgcn1(x, edge_index, edge_type, return_attention)
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn2(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x
