import json
import torch
import argparse
from CounterFactual.Generation import generation

from Loggers.MP_logger import mp_logger
from UnknownFW.load_data import load_mgtab_random
from UnknownFW.load_data import subgraph_hetero_to_homo
from UnknownFW.Graph_Split import Cluster_Split


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--random_seed', type=int, default=114514)
    parser.add_argument('--dataset_seed', type=int, default=42, help='用来 shuffle 数据集, 然后划分 train, val, test')
        
    parser.add_argument('--dataset', type=str, default='mgtab')
    parser.add_argument('--dataset_name', type=str, default='mgtab')
    
    parser.add_argument('--gnn_layers_num', type=int, default=2)
    parser.add_argument('--gnn', type=str, default='GIN')
    
    parser.add_argument('--generation_lr', type=float, default=1e-4)
    parser.add_argument('--pre_lr', type=float, default=1e-4)
    parser.add_argument('--generation_epochs', type=int, default=80)
    parser.add_argument('--subgraph_num', type=int, default=10, help='子图数量')
    parser.add_argument('--pre_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1)            #! 这个 batch_size 影响显存的占用
    parser.add_argument('--temp', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.3)
    
    #! 单独加载某单边同质图
    parser.add_argument('--type_index', type=int, default=7, help='异质图中哪种类型的边')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--device_id', type=int, default=2)
    
    args = parser.parse_args()
    
    vars_args = vars(args)
    print(json.dumps(vars_args, indent=4))
    
    #! 读取原始图
    load_not_split = False
    Original_Graph = load_mgtab_random('/data3/xupin/0_UNName/processed_data_mgtab23/small/', args)
    Original_Graph.node_index=torch.arange(Original_Graph.x.size()[0])
    
    # #! 使用 Cluster 生成子图  
    if load_not_split:
        print('already split')
        SubGraph_List = torch.load('/data3/xupin/0_UNName/data/way_2/'+args.dataset+'_subgraphlist.pt')
    else:
        SubGraph_List = Cluster_Split(Original_Graph, args.subgraph_num)
        torch.save(SubGraph_List,'/data3/xupin/0_UNName/data/way_2/'+args.dataset+'_subgraphlist.pt')
    
    #! 生成 hard negative sample, 并创建 augmented graph
    device = torch.device('cuda:{}'.format(args.device_id))

    #! 分别生成
    for edge_type_index in range(args.type_index):
        #! 创建日志记录器
        main_logger = mp_logger('mgtab_way_2_subgraph_augmented_type_'+str(edge_type_index+1))
        main_logger.info('Augmenting Type '+str(edge_type_index+1)+' Homo SubGraphList......')
        
        #! 分离出每个异质子图的同质边(同质图)
        subgraph_hetero_to_homo_list = subgraph_hetero_to_homo(SubGraph_List, args.dataset, args.type_index)
        nodes_num_list = [each.num_nodes for each in subgraph_hetero_to_homo_list]
        subgraphs_num = len(subgraph_hetero_to_homo_list) 
        
        #! 用同质图的 edge_index, edge_type 替换原异质图中的 edge_index, edge_type
        for subgraph in subgraph_hetero_to_homo_list:
            subgraph.edge_index = subgraph.edge_index_split[edge_type_index]
            subgraph.edge_type = subgraph.edge_type_split[edge_type_index]
        
        print('All heterogeneous subgraphs are replaced by counterpart homo subgraphs')
        main_logger.info('All heterogeneous subgraphs are replaced by counterpart homo subgraphs')
        
        #! 进行 CounterFactual_Augmented
        AugmentedGraph_List = generation(args, subgraphs_num, nodes_num_list, 
                                            subgraph_hetero_to_homo_list[0].node_dim, subgraph_hetero_to_homo_list, 
                                                device, main_logger=main_logger)
        #! 保存增强后的结果
        torch.save(AugmentedGraph_List, '/data3/xupin/0_UNName/data/way_2/'+args.dataset+
                   '_subgraphlist_hetero_to_type_'+str(edge_type_index+1)+'_homo.pt') 

        #! 清除占用
        AugmentedGraph_List=None