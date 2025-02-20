import os
import copy
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.data import Batch
import numpy as np
from sklearn.utils import shuffle

def sample_mask(idx, l):
    """Create mask."""
    mask = torch.zeros(l)
    mask[idx] = 1
    return torch.as_tensor(mask, dtype=torch.bool)

#! 转化为无向图
def relational_undirected(edge_index, edge_type):
    device = edge_index.device
    relation_num = edge_type.max() + 1
    edge_index = edge_index.clone()
    edge_type = edge_type.clone()
    r_edge = []
    for i in range(relation_num):
        e1 = edge_index[:, edge_type == i].unique(dim=1)
        e2 = e1.flip(0)
        edges = torch.cat((e1, e2), dim=1)
        r_edge.append(edges)
    edge_type = torch.cat(
        [torch.tensor([i] * e.shape[1]) for i, e in enumerate(r_edge)],
        dim=0).to(device)
    edge_index = torch.cat(r_edge, dim=1)
    return edge_index, edge_type

def load_mgtab_undirt(pt_raw_data_dir, args):
    '''
        读取原始的处理成 pt 文件的数据 \n
        @Param: pt_raw_data_dir | pt文件存储的文件路径
        
    '''
    path = pt_raw_data_dir
    if os.path.exists('/data3/data/mgtab_undirct.pt'):
        data = torch.load('/data3/data/mgtab_undirct.pt')
        print('Original Graph Already Exsists!')
        print(data)
        return data
    edge_index = torch.load(path + 'edge_index.pt')
    edge_type = torch.load(path + 'edge_type.pt')
    edge_index, edge_type = relational_undirected(edge_index, edge_type)
    x = torch.load(path + 'features.pt')
    label = torch.load(path + 'label.pt')
    
    ## total_node_num = 10199
    total_node_num = x.size()[0]
    
    ## train:val:test = 7:2:1
    train_mask = torch.zeros(total_node_num, dtype=torch.bool)
    train_mask[:int(0.7 * total_node_num)] = True
    val_mask = torch.zeros(total_node_num, dtype=torch.bool)
    val_mask[int(0.7 * total_node_num):int(0.9 * total_node_num)] = True
    test_mask = torch.zeros(total_node_num, dtype=torch.bool)
    test_mask[int(0.9 * total_node_num): total_node_num] = True

    ## 创建一个数据
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type,
                label=label, y=label, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.node_dim = data.x.size()[1]
    
    print(data)
    print('Original Graph Created Successfully!')
    torch.save(data, '/data3/data/'+args.dataset+'_undirct.pt')
    return data

def load_mgtab(pt_raw_data_dir, args):
    '''
        读取原始的处理成 pt 文件的数据 \n
        @Param: pt_raw_data_dir | pt文件存储的文件路径
        
    '''
    path = pt_raw_data_dir
    if os.path.exists('/data3/data/mgtab.pt'):
        data = torch.load('/data3/data/mgtab.pt', map_location='cuda:'+str(args.device_id))
        print('Original Graph Already Exsists!')
        print(data)
        return data
    edge_index = torch.load(path + 'edge_index.pt')
    edge_type = torch.load(path + 'edge_type.pt')
    x = torch.load(path + 'features.pt')
    label = torch.load(path + 'label.pt')
    
    ## total_node_num = 10199
    total_node_num = x.size()[0]
    
    ## train:val:test = 7:2:1
    train_mask = torch.zeros(total_node_num, dtype=torch.bool)
    train_mask[:int(0.7 * total_node_num)] = True
    val_mask = torch.zeros(total_node_num, dtype=torch.bool)
    val_mask[int(0.7 * total_node_num):int(0.9 * total_node_num)] = True
    test_mask = torch.zeros(total_node_num, dtype=torch.bool)
    test_mask[int(0.9 * total_node_num): total_node_num] = True

    ## 创建一个数据
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type,
                label=label, y=label, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.node_dim = data.x.size()[1]
    
    print(data)
    print('Original Graph Created Successfully!')
    torch.save(data, '/data3/data/'+args.dataset+'.pt')
    return data

def load_mgtab_random(pt_raw_data_dir, args):
    '''
        读取原始的处理成 pt 文件的数据 \n
        @Param: pt_raw_data_dir | pt文件存储的文件路径
        
    '''
    path = pt_raw_data_dir
    if os.path.exists('/data3/data/mgtab.pt'):
        data = torch.load('/data3/data/mgtab.pt', map_location='cuda:'+str(args.device_id))
        print('Original Graph Already Exsists!')
        print(data)
        return data
    edge_index = torch.load(path + 'edge_index.pt')
    edge_type = torch.load(path + 'edge_type.pt')
    x = torch.load(path + 'features.pt')
    label = torch.load(path + 'label.pt')
    
    ## total_node_num = 10199
    total_node_num = x.size()[0]
    
    ## train:val:test = 7:2:1
    shuffled_idx = shuffle(np.array(range(total_node_num)), random_state=args.dataset_seed)
    train_idx = shuffled_idx[:int(0.7 * total_node_num)]
    val_idx = shuffled_idx[int(0.7 * total_node_num):int(0.9 * total_node_num)]
    test_idx = shuffled_idx[int(0.9 * total_node_num):]
    train_mask = sample_mask(train_idx, total_node_num)
    val_mask = sample_mask(val_idx, total_node_num)
    test_mask = sample_mask(test_idx, total_node_num)

    ## 创建一个数据
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type,
                label=label, y=label, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.node_dim = data.x.size()[1]
    
    print(data)
    print('Original Graph Created Successfully!')
    torch.save(data, '/data3/data/'+args.dataset+'.pt')
    return data

#! 读取存储为 pt 文件的 twibot_20 原始数据集
def load_pt_raw_data(pt_raw_data_dir):
    '''
        读取原始的处理成 pt 文件的数据 \n
        @Param: pt_raw_data_dir | pt文件存储的文件路径
        
    '''
    path = pt_raw_data_dir
    if os.path.exists('/data3/data/twibot_20_pt_undirct.pt'):
        data = torch.load('/data3/data/twibot_20_pt_undirct.pt')
        return data
        
    edge_index = torch.load(path + 'edge_index.pt')
    edge_type = torch.load(path + 'edge_type.pt')
    edge_index, edge_type = relational_undirected(edge_index, edge_type)
    
    num_relations = edge_type.max() + 1

    #! 创建节点
    x = torch.cat([
        torch.load(path + 'num_properties_tensor.pt'),
        torch.load(path + 'tweets_tensor.pt'),
        torch.load(path + 'cat_properties_tensor.pt'),
        torch.load(path + 'des_tensor.pt')
    ], dim=1)

    #! 创建标签
    label = torch.load(path + 'label.pt')
    total_node_num = x.size()[0]
    label_node_num = label.size()[0]
    sample_idx = list(range(label_node_num))
    
    #! 创建 mask 矩阵
    train_mask = torch.zeros(total_node_num, dtype=torch.bool)
    train_mask[:int(0.7 * label_node_num)] = True
    val_mask = torch.zeros(total_node_num, dtype=torch.bool)
    val_mask[int(0.7 * label_node_num):int(0.9 * label_node_num)] = True
    test_mask = torch.zeros(total_node_num, dtype=torch.bool)
    test_mask[int(0.9 * label_node_num):label_node_num] = True
    
    #! 设置标签: 1-bot, 0-human, -1-unsupervised node (-1表示无标签节点)
    y = torch.cat((label, torch.ones(total_node_num-label_node_num, dtype=torch.int64)), 
                  dim=0)
    y[label_node_num:] = -1     #! 剩余的节点标记为-1
    
    #! 创建 data
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type,
                label=label, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.train_idx = sample_idx[:int(0.7 * label_node_num)]
    data.val_idx = sample_idx[int(0.7 * label_node_num):int(0.9 * label_node_num)]
    data.test_idx = sample_idx[int(0.9 * label_node_num):]
    data.node_dim = data.x.size()[1]
    
    
    print(data)
    torch.save(data, '/data3/data/twibot_20_pt_undirct.pt')
    return data

#! 读取存储为 pt 文件的 twibot_22 原始数据集
def load_pt_raw_data_twibot22(pt_raw_data_dir):
    '''
        读取原始的处理成 pt 文件的数据 \n
        @Param: pt_raw_data_dir | pt文件存储的文件路径
        
    '''
    path = pt_raw_data_dir
    if os.path.exists('/data3/data/twibot_22_pt_undirct.pt'):
        data = torch.load('/data3/data/twibot_22_pt_undirct.pt')
        print('original graph pt already exsist...')
        print(data)
        return data

    raw_data_dir = '/data3/processed_data_twibot22/'
    edge_index = torch.load(raw_data_dir + 'edge_index.pt')
    edge_type = torch.load(raw_data_dir + 'edge_type.pt')
    edge_index, edge_type = relational_undirected(edge_index, edge_type)


    #! 创建节点
    x = torch.load(raw_data_dir+'x.pt')

    #! 设置标签: 1-bot, 0-human, -1-unsupervised node (-1表示无标签节点)
    y = torch.load(raw_data_dir+'label.pt')

    #! 创建标签
    label = torch.load(raw_data_dir + 'label.pt')
    
    #! 直接读取原始数据集中自带数据集分割的预处理为 pt 格式的文件
    train_mask = torch.load(raw_data_dir+'train_mask.pt')
    val_mask = torch.load(raw_data_dir+'val_mask.pt')
    test_mask = torch.load(raw_data_dir+'test_mask.pt')
    
    #! 创建 data
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type,
                label=label, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.node_dim = data.x.size()[1]
    
    print('original graph pt create successfully!')
    print(data)
    torch.save(data, '/data3/data/twibot_22_pt_undirct.pt')
    return data

#! 读取存储为 pt 文件的 twibot_22_100w 原始数据集
def load_pt_raw_data_twibot22_100w():
    '''
        读取原始的处理成 pt 文件的数据 \n
        @Param: pt_raw_data_dir | pt文件存储的文件路径
        
    '''
    if os.path.exists('/data3/data/twibot_22_100w_pt_undirct.pt'):
        data = torch.load('/data3/data/twibot_22_100w_pt_undirct.pt')
        print('original graph pt already  xsist...')
        print(data)
        return data

    raw_data_dir = '/data3/processed__raw_twibot_22/data/all_100w_nodes/processed/graph_component/'
    edge_index = torch.load(raw_data_dir + 'edge_index.pt')
    edge_type = torch.load(raw_data_dir + 'edge_type.pt')
    edge_index, edge_type = relational_undirected(edge_index, edge_type)


    #! 创建节点
    x = torch.load(raw_data_dir+'x.pt')

    #! 设置标签: 1-bot, 0-human, -1-unsupervised node (-1表示无标签节点)
    y = torch.load(raw_data_dir+'label.pt')

    #! 创建标签
    label = torch.load(raw_data_dir + 'label.pt')
    
    #! 直接读取原始数据集中自带数据集分割的预处理为 pt 格式的文件
    train_mask = torch.load(raw_data_dir+'train_mask.pt')
    val_mask = torch.load(raw_data_dir+'val_mask.pt')
    test_mask = torch.load(raw_data_dir+'test_mask.pt')
    
    #! 创建 data
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type,
                label=label, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.node_dim = data.x.size()[1]
    
    print('original graph pt create successfully!')
    print(data)
    torch.save(data, '/data3/data/twibot_22_100w_pt_undirct.pt')
    return data

#! 读取存储为 pt 文件的 twibot_22_100w 原始数据集
def load_pt_raw_data_twibot22_100w_xlm():
    '''
        读取原始的处理成 pt 文件的数据 \n
        @Param: pt_raw_data_dir | pt文件存储的文件路径
        
    '''
    if os.path.exists('/data3/data/twibot_22_100w_xlm_undirct.pt'):
        data = torch.load('/data3/data/twibot_22_100w_xlm_undirct.pt')
        print('original graph pt already  xsist...')
        print(data)
        return data

    raw_data_dir = '/data3/processed__raw_twibot_22/data/all_100w_nodes_xlm/processed/graph_component/'
    edge_index = torch.load(raw_data_dir + 'edge_index.pt')
    edge_type = torch.load(raw_data_dir + 'edge_type.pt')
    edge_index, edge_type = relational_undirected(edge_index, edge_type)


    #! 创建节点
    x = torch.load(raw_data_dir+'x.pt')

    #! 设置标签: 1-bot, 0-human, -1-unsupervised node (-1表示无标签节点)
    y = torch.load(raw_data_dir+'label.pt')

    #! 创建标签
    label = torch.load(raw_data_dir + 'label.pt')
    
    #! 直接读取原始数据集中自带数据集分割的预处理为 pt 格式的文件
    train_mask = torch.load(raw_data_dir+'train_mask.pt')
    val_mask = torch.load(raw_data_dir+'val_mask.pt')
    test_mask = torch.load(raw_data_dir+'test_mask.pt')
    
    #! 创建 data
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type,
                label=label, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.node_dim = data.x.size()[1]
    
    print('original graph pt create successfully!')
    print(data)
    torch.save(data, '/data3/data/twibot_22_100w_xlm_undirct.pt')
    return data

#! 将多边异质图转化为多个同质图 (hetero_to_homo)
def hetero_to_homo(pt_data_dir):
    '''
        读取原始的处理成 pt 文件的数据 \n
        @Param: pt_data_dir | pt文件存储的文件路径
        
    '''
    if (os.path.exists(pt_data_dir+'twibot_22_type_1_pt_undirct.pt')) and (os.path.exists(pt_data_dir+'twibot_22_type_2_pt_undirct.pt')):
        Type_1_Original_Graph = torch.load(pt_data_dir+'twibot_22_type_1_pt_undirct.pt')
        print(Type_1_Original_Graph)
        print()

        Type_2_Original_Graph = torch.load(pt_data_dir+'twibot_22_type_2_pt_undirct.pt')
        print(Type_2_Original_Graph)
        print()
        return [Type_1_Original_Graph, Type_2_Original_Graph]
        
    Original_Hetero_Graph = torch.load(pt_data_dir+'twibot_22_pt_undirct.pt')
    edge_type =Original_Hetero_Graph.edge_type
    edge_index = Original_Hetero_Graph.edge_index
    print(Original_Hetero_Graph)
    print()

    print('creating edge type_1 homo graph...')
    #! 表示 type 为 0 的边,
    edge_indices_0 = torch.nonzero(edge_type == 0).T
    source_nodes = edge_index[0, edge_indices_0][0]
    target_nodes = edge_index[1, edge_indices_0][0]
    edge_type_0_edges = torch.stack((source_nodes, target_nodes), dim=0)
    print(edge_type_0_edges.size())

    #! 分离类型一的同质图
    Type_1_Original_Graph = copy.deepcopy(Original_Hetero_Graph)
    Type_1_Original_Graph.edge_type = edge_indices_0[0]
    Type_1_Original_Graph.edge_index = edge_type_0_edges
    torch.save(Type_1_Original_Graph, '/data3/data/twibot_22_type_1_pt_undirct.pt')
    print(Type_1_Original_Graph)
    print()

    print('creating edge type_2 homo graph...')
    #! 表示 type 为 1 的边
    edge_indices_1 = torch.nonzero(edge_type == 1).T
    source_nodes = edge_index[0, edge_indices_1][0]
    target_nodes = edge_index[1, edge_indices_1][0]
    edge_type_1_edges = torch.stack((source_nodes, target_nodes), dim=0)
    print(edge_type_1_edges.size())

    #! 分离类型二的同质图
    Type_2_Original_Graph = copy.deepcopy(Original_Hetero_Graph)
    Type_2_Original_Graph.edge_type = edge_indices_1[0]         #! 这里要修改一下, 为1或0
    Type_2_Original_Graph.edge_index = edge_type_1_edges
    torch.save(Type_2_Original_Graph, '/data3/data/twibot_22_type_2_pt_undirct.pt')
    print(Type_2_Original_Graph)
    print()
    return [Type_1_Original_Graph, Type_2_Original_Graph]

#! 子异质图转化为多个同质图 (subgraph_hetero_to_homo)
def subgraph_hetero_to_homo(subgraph_list, dataset_name, type_num, save_dir='/data3/data/way_2/'):
    '''
        @Param: subgraph_list | 子图列表 \n
        @Param: save_dir | 结果保存路径
    '''
    #! 如果处理过了, 就返回
    if os.path.exists(save_dir+dataset_name+'_subgraphlist_hetero_to_homo.pt'):
        print('already exists...')
        subgraph_hetero_to_homo_list = torch.load(save_dir+dataset_name+'_subgraphlist_hetero_to_homo.pt')
        return subgraph_hetero_to_homo_list
    #! 从子异质图中分离出同质图
    #! 用来记录分离结果
    subgraph_hetero_to_homo_list = []
    for subgraph in tqdm(subgraph_list, total=len(subgraph_list), desc='Converting Hetero SubGraph into Homo... '):
        edge_type = subgraph.edge_type
        edge_index = subgraph.edge_index
        subgraph.edge_type_split = []
        subgraph.edge_index_split = []
        ## 这里要面临一个问题: 如果这个子图没有边会怎么办? | 解决办法应该是重新构图
        for type_index in range(type_num):
            #! num_edges 为边的数量
            _, num_edges = edge_index.shape
            #! 如果整个子图都是孤立节点
            if num_edges==0:
                #! 创建空的 edge_index_tensor 和 edge_type_tenosr
                edge_type_split_save = torch.tensor([], dtype=torch.int64)
                edge_index_split_save = torch.tensor([[],[]], dtype=torch.int64)
                #! 记录结果
                subgraph.edge_type_split.append(edge_type_split_save)
                subgraph.edge_index_split.append(edge_index_split_save)                
            else:
                #! 获得对应类型边的索引
                edge_indices= torch.nonzero(edge_type == type_index).T
                source_nodes = edge_index[0, edge_indices][0]
                target_nodes = edge_index[1, edge_indices][0]
                edge_type_edges = torch.stack((source_nodes, target_nodes), dim=0)
                #! 分离出同质图
                subgraph.edge_type_split.append(edge_indices[0])
                subgraph.edge_index_split.append(edge_type_edges)
        subgraph_hetero_to_homo_list.append(subgraph)

    #! 保存结果
    torch.save(subgraph_hetero_to_homo_list,save_dir+dataset_name+'_subgraphlist_hetero_to_homo.pt')
    return subgraph_hetero_to_homo_list

#! 合并增强后的子图列表
def merge_augmented_subgraphlist(pt_data_dir, original_ptgraph_dir, edge_list=None):
    '''
        合并生成的增强子图列表
        读取原始的处理成 pt 文件的数据 \n
        @Param: pt_data_dir | 增强子图存放的文件路径 \n
        @Param: original_ptgraph_dir | 原始pt形式图存放的文件路径 \n
        @Param: edge_type_list | 边类型列表
    '''
    merged_type_graph_list = []
    edge_type_list = []
    edge_index_list = []
    
    index=0
    #! 读取子图, 将其合并为大图
    for edge_type in edge_list:
        print('handing ', edge_type)
        augmented_subgraph_list = torch.load(pt_data_dir+'twibot_22_'+edge_type+'_pt_undirct_augmented_subgraph.pt')
        merged_graph = Batch.from_data_list(augmented_subgraph_list)
        
        #! 创建 merged 后的大图
        node_dim=merged_graph.node_dim[0]
        merged_graph = Data(x=merged_graph.x, edge_index=merged_graph.edge_index, y=merged_graph.y,
                                train_mask=merged_graph.train_mask, val_mask=merged_graph.val_mask, test_mask=merged_graph.test_mask)
        merged_graph.node_dim=node_dim
        merged_graph.edge_type=torch.ones(merged_graph.edge_index.size()[1],dtype=torch.int64)*index

        index_list = []
        for subgraph in augmented_subgraph_list:
            index_list.append(subgraph.node_index)
        merged_index = torch.cat(index_list, dim=0)
        merged_graph.merged_index=merged_index
        
        #! 对merged_graph 进行顺序恢复, 将无序的node和edge_index恢复到原来的, 以便两个同质图合并为一个异质图
        #! 恢复 data.x
        recovered_index = torch.argsort(merged_graph.merged_index)  
        node_recovered = merged_graph.x[recovered_index]
        merged_graph.x = node_recovered
        print('data.x recovered successfully')
                
        #! 恢复 data.edge_index
        reflection_dict = {}
        for new, old in zip(torch.arange(merged_graph.x.size()[0]), merged_index):
            reflection_dict[new.item()]=old.item()
        new_edge_index = merged_graph.edge_index

        recovered_edge_index = []
        for row in new_edge_index:
            row_list = []
            #! 这里比较慢
            for i in range(row.size()[0]):
                row_list.append(reflection_dict[row[i].item()])
            recovered_edge_index.append(row_list)
        merged_graph.edge_index = torch.tensor(recovered_edge_index)
        print('data.edge_index recovered successfully')
        
        #! 恢复 data.y
        label = torch.load(original_ptgraph_dir + 'label.pt')
        total_node_num = merged_graph.x.size()[0]
        label_node_num = label.size()[0]
        y = torch.cat((label, torch.ones(total_node_num-label_node_num, dtype=torch.int64)), dim=0)         #! 设置标签: 1-bot, 0-human, -1-unsupervised node (-1表示无标签节点)
        y[label_node_num:] = -1    
        merged_graph.y = y
        print('data.y recovered successfully')
        
        #! 恢复 train_mask, val_mask, test_mask
        train_mask = torch.zeros(total_node_num, dtype=torch.bool)
        train_mask[:int(0.7 * label_node_num)] = True
        val_mask = torch.zeros(total_node_num, dtype=torch.bool)
        val_mask[int(0.7 * label_node_num):int(0.9 * label_node_num)] = True
        test_mask = torch.zeros(total_node_num, dtype=torch.bool)
        test_mask[int(0.9 * label_node_num):label_node_num] = True
        merged_graph.train_mask = train_mask
        merged_graph.val_mask = val_mask
        merged_graph.test_mask = test_mask
        
        #! 保存增强子图列表合并后的图, 两个图的顺序乱了
        torch.save(merged_graph, pt_data_dir+'twibot_22_'+edge_type+'_pt_undirct_augmented_subgraph_merged.pt')
        index+=1
    
#! 合并两个 homo graph, 生成 hetero_graph
def homo_to_hetero(pt_merged_data_dir, edge_type_list):
    '''
        将两个 homo_graph 合并为 hetero_graph
        @Param: pt_merged_data_dir | 增强子图存放的文件路径 \n
        @Param: edge_type_list | 边类型列表
    '''
    index=0
    #! 将 homo graph 的 edge_index, edge_type 进行合并
    merged_graph_list = []
    merged_edge_index_list = []
    merged_edge_type_list = []
    for edge_type in edge_type_list:
        merged_graph = torch.load(pt_merged_data_dir+'twibot_22_'+edge_type+'_pt_undirct_augmented_subgraph_merged.pt')
        merged_graph_list.append(merged_graph)
        merged_edge_index_list.append(merged_graph.edge_index)
        merged_edge_type_list.append(merged_graph.edge_type)        
        
    for index, edge_type in enumerate(edge_type_list):
        merged_graph = merged_graph_list[index]
        merged_graph.edge_index = torch.cat(merged_edge_index_list, dim=1)
        merged_graph.edge_type = torch.cat(merged_edge_type_list, dim=0)
        torch.save(merged_graph, pt_merged_data_dir+'twibot_22_'+edge_type+'_node_pt_undirct_hetero.pt')

#! 合并两个 homo subgraph, 生成 hetero_subgraph
def subgraph_homo_to_hetero(pt_data_dir, edge_type_list, dataset_name='twibot_20'):
    '''
        将两个 sub homo_graph 合并为 sub hetero_graph
        针对的子图级别的处理
        @Param: pt_data_dir | 增强子图存放的文件路径 \n
        @Param: edge_type_list | 边类型列表
    '''
    if (os.path.exists('/data3/data/way_2/'+dataset_name+'_subgraphlist_hetero_to_type_1_node_hetero.pt')) and \
        (os.path.exists('/data3/data/way_2/'+dataset_name+'_subgraphlist_hetero_to_type_2_node_hetero.pt')):
            print('already converted...')
            return
    type_1_homo_subgraph_list = torch.load('/data3/data/way_2/'+dataset_name+'_subgraphlist_hetero_to_type_1_homo.pt')
    type_2_homo_subgraph_list = torch.load('/data3/data/way_2/'+dataset_name+'_subgraphlist_hetero_to_type_2_homo.pt')
    for type_1_subgraph, type_2_subgraph in zip(type_1_homo_subgraph_list, type_2_homo_subgraph_list):
        #! 合并 edge_index
        edge_index_merged = []
        edge_index_merged.append(type_1_subgraph.edge_index)
        edge_index_merged.append(type_2_subgraph.edge_index)
        #! 合并 edge_type
        edge_type_merged = []
        edge_type_merged.append(torch.zeros(type_1_subgraph.edge_index.size()[1], dtype=torch.int64))
        edge_type_merged.append(torch.ones(type_2_subgraph.edge_index.size()[1], dtype=torch.int64))
        #! 进行 torch.cat() 就可实现
        edge_index_merged = torch.cat(edge_index_merged, dim=1)
        edge_type_merged = torch.cat(edge_type_merged, dim=0)
        #! 用merged后的 edge_index, edge_type 替换原来的 type_1, type_2 graph
        type_1_subgraph.edge_index = edge_index_merged
        type_2_subgraph.edge_index = edge_index_merged
        type_1_subgraph.edge_type = edge_type_merged
        type_2_subgraph.edge_type = edge_type_merged
    torch.save(type_1_homo_subgraph_list, '/data3/data/way_2/'+dataset_name+'_subgraphlist_hetero_to_type_1_node_hetero.pt')
    torch.save(type_2_homo_subgraph_list, '/data3/data/way_2/'+dataset_name+'_subgraphlist_hetero_to_type_2_node_hetero.pt')
    print('converted successfully')

def mgtab_subgraph_homo_to_hetero(pt_data_dir, type_num, dataset_name='twibot_20'):
    if os.path.exists(pt_data_dir+dataset_name+'_subgraphlist_augmented.pt'):
        print('already converted...')
        return
    
    homo_subgraph_list = torch.load(pt_data_dir+dataset_name+'_subgraphlist_hetero_to_type_1_homo.pt')

    for graph_index, subgraph in enumerate(homo_subgraph_list):
        edge_index_merged = []
        edge_type_merged = []
        edge_index_merged.append(subgraph.edge_index)
        edge_type_merged.append(torch.zeros(subgraph.edge_index.size()[1], dtype=torch.int64))
        
        subgraph.edge_index_split[0]=subgraph.edge_index
        for type_index in range(1, type_num):
            #! 合并 edge_index
            subgraph_list = torch.load(pt_data_dir+dataset_name+'_subgraphlist_hetero_to_type_'+str(type_index)+'_homo.pt')
            edge_index_merged.append(subgraph_list[graph_index].edge_index)
            subgraph.edge_index_split[type_index] = subgraph_list[graph_index].edge_index
            
            #! 合并 edge_type
            edge_type_merged.append(torch.ones(subgraph_list[graph_index].edge_index.size()[1], dtype=torch.int64)*type_index)
            
            #! 删除变量
            del subgraph_list
        #! 进行 torch.cat() 就可实现
        edge_index_merged = torch.cat(edge_index_merged, dim=1)
        edge_type_merged = torch.cat(edge_type_merged, dim=0)
        #! 用merged后的 edge_index, edge_type 替换原来的 type_1, type_2 graph
        subgraph.edge_index = edge_index_merged
        subgraph.edge_type = edge_type_merged
    
    torch.save(homo_subgraph_list, pt_data_dir+dataset_name+'_subgraphlist_augmented.pt')    
    print('converted successfully')
    
#! Way_2 中用来读取子图列表
def load_subgraph_list(subgraphlist_dir):
    '''
        读取 Graph_Split 后的子图列表
        @Param: subgraphlist_dir | 子图列表存放的位置
    '''
    subgraph_list = torch.load(subgraphlist_dir+'')
