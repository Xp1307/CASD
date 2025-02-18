from torch import nn
from torch.nn import ParameterList
from torch.nn.parameter import Parameter
import torch
from torch_geometric.nn import GCNConv, GINConv
from torch_scatter import scatter_mean
import copy


class GNN(nn.Module):
    def __init__(self, args, attrs_dim):
        super(GNN, self).__init__()

        self.args = args
        if args.gnn == 'GCN':
            self.gnn_layers = nn.ModuleList([GCNConv(attrs_dim, attrs_dim) for i in range(args.gnn_layers_num)])
        if args.gnn == 'GIN':
            self.gnn_layers = nn.ModuleList([GINConv(MLP(attrs_dim, [2*attrs_dim, 2*attrs_dim, attrs_dim])) for i in range(args.gnn_layers_num)])
        self.activation = nn.Tanh()

    def forward(self, data):
        x = data.x.float()
        for i in range(self.args.gnn_layers_num):
            x = self.gnn_layers[i](x, data.edge_index)
            x = self.activation(x)
        x = scatter_mean(x, data.batch, dim=0)
        return x


class MLP(nn.Module):
    #! 16,8,2 这儿的信息是不是压缩得太狠了, 是的表示没办法区分开来
    #! 128, 32, 2, 改成这个试试?
    def __init__(self, attrs_dim, dim_list=[16, 8, 2]):
        super(MLP, self).__init__()

        attrs_dim = [attrs_dim]
        attrs_dim.extend(dim_list)
        self.layers = nn.ModuleList([nn.Linear(attrs_dim[i], attrs_dim[i+1]) for i in range(len(dim_list))])
        self.activation = nn.Tanh()

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.activation(x)
        return x


#! 这里的 attrs_dim 是 1544
class Predictor(nn.Module):
    def __init__(self, args, attrs_dim):
        super(Predictor, self).__init__()

        self.gnn = GNN(args, attrs_dim)
        self.mlp = MLP(attrs_dim)
        # self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, data):
        x = self.gnn(data)
        graph_embedding = x
        #@ 我觉得这里可以改一改, 数据维度压缩得太狠了, 可以改一改维度
        x = self.mlp(x)
        # x = self.logsoftmax(x)
        return x, graph_embedding


class Generator(nn.Module):
    def __init__(self, args, graphs_num, nodes_num_list, attrs_dim):
        super(Generator, self).__init__()

        self.args = args
        self.attrs_dim = attrs_dim
        self.predictor = Predictor(args, attrs_dim)

        '''
            这里就需要考虑到 edge_type 同等的变化, 这个变化应该随着 edge_index 的变化而变化
        '''
        #! 边扰动矩阵 (邻接关系)
        self.perturbation_matrices = ParameterList([Parameter(torch.FloatTensor(nodes_num_list[i], nodes_num_list[i])) for i in range(graphs_num)])
        for each in self.perturbation_matrices:
            #print(each.data)
            torch.nn.init.xavier_uniform_(each.data, gain=5/3)
        
        #! 偏差项
        self.perturbation_biases = ParameterList([Parameter(torch.FloatTensor(nodes_num_list[i], nodes_num_list[i])) for i in range(graphs_num)])
        for each in self.perturbation_biases:
            torch.nn.init.xavier_uniform_(each.data, gain=5/3)

        #! 节点特征矩阵
        self.masking_matrices = ParameterList([Parameter(torch.FloatTensor(nodes_num_list[i], attrs_dim)) for i in range(graphs_num)])
        for each in self.masking_matrices:
            torch.nn.init.xavier_uniform_(each.data, gain=5/3)

    #! 根据 batch.id 进行的训练, 选取对应的矩阵
    def forward(self, batch, device):
        '''
            @Param: batch | 表示批次训练数据
            @Param: device | 表示使用的训练设备(CPU or GPU)
        '''
        #! 下面表示对 batch 中的多个子图进行拼接, 邻接扰动是一个对角矩阵
        perturbation_matrices = tuple([self.perturbation_matrices[id] for id in batch.id])
        perturbation_matrices = torch.block_diag(*perturbation_matrices)

        perturbation_biases = tuple([self.perturbation_biases[id] for id in batch.id])
        perturbation_biases = torch.block_diag(*perturbation_biases)

        masking_matrices = [self.masking_matrices[int(id)] for id in batch.id]
        masking_matrices = torch.cat(masking_matrices, dim=0)

        #! 只扰动边
        batch_perturbation = copy.deepcopy(batch)
        #! 只扰动矩阵
        batch_masking = copy.deepcopy(batch)

        #! 下面表示对边进行扰动
        values = torch.Tensor([1 for i in range(batch.edge_index.size()[1])])
        if self.args.cuda:
            values = values.to(device)
        adjs = torch.sparse_coo_tensor(batch.edge_index, values, (batch.num_nodes, batch.num_nodes), dtype=torch.float)
        adjs_dense = adjs.to_dense()
        perturbation_adjs = torch.mm(perturbation_matrices, adjs_dense)+perturbation_biases
        perturbation_adjs = torch.sigmoid(perturbation_adjs)
        #! 这里使对 edge_index 进行扰动
        perturbation_adjs = torch.where(perturbation_adjs<=0.5, torch.zeros_like(perturbation_adjs), torch.ones_like(perturbation_adjs))
        perturbation_adjs_sparse = perturbation_adjs.to_sparse()
        batch_perturbation.edge_index = perturbation_adjs_sparse.indices()

        #! 下面是进行节点属性的增强
        #! batch[0].node_dim 表示节点的特征维度
        masking_matrices = torch.sigmoid(masking_matrices)
        masking_matrices = torch.where(masking_matrices<=0.5, torch.zeros_like(masking_matrices), torch.ones_like(masking_matrices))
        masked_attrs = torch.mul(masking_matrices, batch.x)
        batch_masking.x = masked_attrs
        
        #! 这里的 Predictor 其实就是先过一个 GNN, 再过一个 MLP
        predicted_results, _ = self.predictor(batch)
        
        #! 这里的 perturbation_predicted_results, 只有邻接关系被扰动, 节点属性没有被增强
        #! 同理, masking_predicted_results, 只有节点属性被增强, 邻接关系没有被扰动
        perturbation_predicted_results, _ = self.predictor(batch_perturbation)
        masking_predicted_results, _ = self.predictor(batch_masking)

        return adjs_dense, perturbation_adjs, masking_matrices, predicted_results, perturbation_predicted_results, masking_predicted_results, batch_perturbation, batch_masking