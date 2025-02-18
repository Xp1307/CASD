import random
import numpy as np
import torch
import torch_geometric
# from torch_geometric.data import ClusterData
from torch_geometric.data import Batch
from torch_geometric.loader import ClusterData, ClusterLoader

def Cluster_Split(Original_Graph, SubGraph_Num=100):
    '''
        输入为原始图, 输出为划分后的子图列表 \n
        @Param: Original_Graph - 原始图 \n
        @Param: SubGraph_Num - 子图数量 
    '''
    cluster_data = ClusterData(Original_Graph, num_parts=SubGraph_Num, recursive=False)
    SubGraph_List = list(cluster_data)
        
    for index, SubGraph in enumerate(SubGraph_List):
        SubGraph.id = index
        SubGraph.node_dim = Original_Graph.node_dim

    return SubGraph_List

# def Cluster_Split_Shuffle(Original_Graph, SubGraph_Num=100):
#     '''
#         输入为原始图, 输出为划分后的子图列表 \n
#         @Param: Original_Graph - 原始图 \n
#         @Param: SubGraph_Num - 子图数量 
#     '''
#     cluster_data = ClusterData(Original_Graph, num_parts=3200)  # 将图划分为 1500 个子图
#     SubGraph_List = list(cluster_data)
        
#     for index, SubGraph in enumerate(SubGraph_List):
#         SubGraph.id = index
#         SubGraph.node_dim = Original_Graph.node_dim

#     return SubGraph_List

def Cluster_Split_Shuffle(Original_Graph, SubGraph_Num=100):
    '''
        输入为原始图, 输出为划分后的子图列表 \n
        @Param: Original_Graph - 原始图 \n
        @Param: SubGraph_Num - 子图数量 
    '''
    seed = 9
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cluster_data = ClusterData(Original_Graph, num_parts=3200)  # 将图划分为 1500 个子图
    train_loader = ClusterLoader(cluster_data, batch_size=4, shuffle=True)

    SubGraph_List = list(train_loader)
        
    for index, SubGraph in enumerate(SubGraph_List):
        SubGraph.id = index
        SubGraph.node_dim = Original_Graph.node_dim
    return SubGraph_List