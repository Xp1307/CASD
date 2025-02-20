import os
import time
import argparse

import torch

from UnknownFW.load_data import load_pt_raw_data
from UnknownFW.load_data import hetero_to_homo
from UnknownFW.Graph_Split import Cluster_Split
from CounterFactual.Generation import generation
from Loggers.MP_logger import mp_logger

#! 创建两个进程
import multiprocessing
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', type=bool, default=True)
    
    parser.add_argument('--random_seed', type=int, default=114514)
    parser.add_argument('--dataset', type=str, default='Twibot-20')
    parser.add_argument('--gnn_layers_num', type=int, default=2)
    parser.add_argument('--gnn', type=str, default='GIN')
    parser.add_argument('--generation_lr', type=float, default=1e-4)
    parser.add_argument('--pre_lr', type=float, default=1e-4)
    parser.add_argument('--generation_epochs', type=int, default=80)
    parser.add_argument('--subgraph_num', type=int, default=200, help='子图数量')
    parser.add_argument('--pre_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1)            #! 这个 batch_size 影响显存的占用
    parser.add_argument('--temp', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.3)
    
    #! 单独加载某单边同质图
    parser.add_argument('--type_index', type=int, default=2, help='异质图中哪种类型的边')
    parser.add_argument('--device_id', type=int, default=1)
    
    #! 使用多进程 ——— 将异质图转化为多个同质图
    parser.add_argument('--is_multiprocess', type=bool, default=False, help='是否使用多进程')
    parser.add_argument('--multiprocess_num', type=int, default=2, help='进程数量, 和数据集有关')
    parser.add_argument('--is_hetero', type=bool, default=False, help='直接使用异质图还是同质图')
       
    args = parser.parse_args()
    
    #! 如果使用多进程
    if args.is_multiprocess:
        #! 使用原始的异质图
        if args.is_hetero:
            #! 创建日志记录器
            main_logger = mp_logger('main')
            
            #! 读取数据并划分子图
            data_dir_path = '/data3/processed_data/'
            Original_Graph=load_pt_raw_data(data_dir_path)
            SubGraph_List = Cluster_Split(Original_Graph, args.subgraph_num)
            subgraphs_num = len(SubGraph_List)
            nodes_num_list = [each.num_nodes for each in SubGraph_List]
            
            #! 确定使用的 GPU
            device = torch.device('cuda:{}'.format(args.device_id))
            
            #! 生成 hard negative sample, 并创建 augmented graph
            AugmentedGraph_List = generation(args, subgraphs_num, nodes_num_list, Original_Graph.node_dim, SubGraph_List, device, main_logger)
            torch.save(AugmentedGraph_List, '/data3/data/AugmentedSubGraph_List.pt')

        #! 将异质图转化为单边同质图
        else:
            Original_Graph_Path = '/data3/data/'
            #! 分别所有单边同质图
            Type_Original_Graph_List = hetero_to_homo(Original_Graph_Path)
            
            #! 创建一个 Queue 对象
            result_queue = multiprocessing.Queue()
            multiprocess_list = []

            for index in range(args.multiprocess_num):
                Type_Original_Graph = Type_Original_Graph_List[index]
                #! 划分子图
                SubGraph_List = Cluster_Split(Type_Original_Graph, args.subgraph_num)
                subgraphs_num = len(SubGraph_List)
                nodes_num_list = [each.num_nodes for each in SubGraph_List]
            
                #! 确定使用的 GPU
                device = torch.device('cuda:{}'.format(index))
                mp_data_name = 'type_'+str(index)
                func_args = (args, subgraphs_num, nodes_num_list, Type_Original_Graph.node_dim, SubGraph_List, device, None,
                            mp_data_name, result_queue)
                # 创建进程，并传递参数
                process = multiprocessing.Process(name='Process_1'+str(index), target=generation, args=func_args)
                multiprocess_list.append(process)
            
            # 启动进程
            for index in range(args.multiprocess_num):
                multiprocess_list[index].start()

            # 等待进程完成           
            for index in range(args.multiprocess_num):
                multiprocess_list[index].join()      

            # 从队列中获取结果
            results = []
            while not result_queue.empty():
                print(result_queue.get())
                results.append(result_queue.get())
            
            print("Results from both processes:", results)
    else:
        #! 使用原始的异质图
        if args.is_hetero:
            #! 创建日志记录器
            main_logger = mp_logger('main')
            
            #! 读取数据并划分子图
            data_dir_path = '/data3/processed_data/'
            Original_Graph=load_pt_raw_data(data_dir_path)
            SubGraph_List = Cluster_Split(Original_Graph, args.subgraph_num)
            subgraphs_num = len(SubGraph_List)
            nodes_num_list = [each.num_nodes for each in SubGraph_List]
            
            #! 确定使用的 GPU
            device = torch.device('cuda:{}'.format(args.device_id))
            
            #! 生成 hard negative sample, 并创建 augmented graph
            AugmentedGraph_List = generation(args, subgraphs_num, nodes_num_list, Original_Graph.node_dim, SubGraph_List, device, main_logger)
            torch.save(AugmentedGraph_List, '/data3/data/AugmentedSubGraph_List.pt')
        else:
            #! 创建日志记录器
            main_logger = mp_logger('type'+'_'+str(args.type_index))
            
            #! 读取数据并划分子图
            data_dir_path = '/data3/data/'
            Original_Graph = torch.load(data_dir_path+'twibot_20_type_'+str(args.type_index)+'_pt_undirct.pt')
            Original_Graph.node_index=torch.arange(Original_Graph.x.size()[0])  ## 添加一个 node_index, 用来记录节点的索引值                                    
            
            SubGraph_List = Cluster_Split(Original_Graph, args.subgraph_num)
            subgraphs_num = len(SubGraph_List)
            nodes_num_list = [each.num_nodes for each in SubGraph_List]
            
            #! 确定使用的 GPU
            device = torch.device('cuda:{}'.format(args.device_id))
            
            #! 生成 hard negative sample, 并创建 augmented graph
            AugmentedGraph_List = generation(args, subgraphs_num, nodes_num_list, Original_Graph.node_dim, SubGraph_List, device, main_logger=main_logger)
            torch.save(AugmentedGraph_List, '/data3/data/twibot_20_type_'+str(args.type_index)+'_pt_undirct_augmented_subgraph'+'.pt')          
