import os
import argparse

from UnknownFW.load_data import load_pt_raw_data
from UnknownFW.Graph_Split import Cluster_Split
from CounterFactual.DP_Generation import generation
from Loggers.MP_logger import mp_logger

#! 多 GPU 并行
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Queue                               # 记录子进程的返回值
from torch.nn.parallel import DistributedDataParallel as DDP

#! 设置分布式环境变量
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '11451'

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
    
    #! 创建日志记录器
    main_logger = mp_logger('main')
    
    #! 使用数据并行计算参数
    parser.add_argument('--is_data_distributed', type=bool, default=False, help='是否使用分布式并行计算')
    parser.add_argument('--world_size', type=int, default=2, help='使用的 GPU 设备数量')
    
    args = parser.parse_args()
    
    #! 读取数据并划分子图
    data_dir_path = '/data3/xupin/0_UNName/processed_data/'
    Original_Graph=load_pt_raw_data(data_dir_path)
    SubGraph_List = Cluster_Split(Original_Graph, args.subgraph_num)
    subgraphs_num = len(SubGraph_List)
    nodes_num_list = [each.num_nodes for each in SubGraph_List]
    
    #! 使用多卡 GPU 进行数据并行操作
    world_size = args.world_size
    return_queue = Queue()
    lock = mp.Lock()  # 创建一个锁
    mp.spawn(generation, args=(world_size, args, subgraphs_num, nodes_num_list, Original_Graph.node_dim, SubGraph_List, return_queue, lock), 
             nprocs=world_size, join=True)
    torch.save(return_queue, '/data3/xupin/0_UNName/data/AugmentedSubGraph_List.pt')