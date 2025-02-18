import os
import copy
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader

#! 访问当前目录下的文件
from .Generator import Generator
#! 用来创建日志记录器
from Loggers.MP_logger import mp_logger

#! 多 GPU 并行
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def similarity_loss(adjs_dense, perturbation_adjs, masking_matrix):
    diff = adjs_dense-perturbation_adjs
    diff_norm = torch.linalg.matrix_norm(diff, ord=1)/torch.ones_like(diff).sum()
    masking_matrix_norm = torch.linalg.matrix_norm(masking_matrix, ord=1)/torch.ones_like(masking_matrix).sum()
    return diff_norm-masking_matrix_norm

def kl_div(predicted_results, perturbation_predicted_results, masking_predicted_results):
    predicted_results = predicted_results.softmax(dim=1)
    perturbation_predicted_results = perturbation_predicted_results.log_softmax(dim=1)
    masking_predicted_results = masking_predicted_results.log_softmax(dim=1)
    loss_func = torch.nn.KLDivLoss(reduction='batchmean')
    kl_loss_1 = loss_func(perturbation_predicted_results, predicted_results)
    kl_loss_2 = loss_func(masking_predicted_results, predicted_results)
    return kl_loss_1+kl_loss_2


def generation(rank, world_size, args, subgraphs_num, nodes_num_list, node_attrs_dim, subgraph_list, return_queue, lock):
    '''
        @Param: args | 参数列表 \n
        @Param: subgraphs_num | 子图数量 \n
        @Param: nodes_num_list | 子图节点数量列表 \n
        @Param: node_attrs_dim | 节点特征维度 \n
        @Param: subgraph_list | 子图列表
        @Param: return_queue | 用来记录每个进程的返回值
        @Param: lock | 创建的锁
    '''
    #! 获得当前函数运行的进程 ID
    current_pid = os.getpid()
    
    #! 创建日志记录器
    Current_logger = mp_logger(str(current_pid))
    Current_logger.info('PID:{}, Hard Negative Samples Generation Start...'.format(current_pid))

    #! 计算每个进程需要处理的子图数量, 让每个进程只处理自己负责的部分
    num_subgraphs_per_process = subgraphs_num // world_size
    start_index = rank * num_subgraphs_per_process
    end_index = (rank + 1) * num_subgraphs_per_process if rank != world_size - 1 else subgraphs_num
    current_subgraph_list = subgraph_list[start_index:end_index]
    
    #! 初始化并行进程, 并为每个进程选择一个 cuda
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device('cuda:{}'.format(rank))
      
    #! 为每个进程创建 dataloader
    dataloader = DataLoader(current_subgraph_list, batch_size=args.batch_size, shuffle=True)
    dataloader = [batch for batch in dataloader]
    if args.cuda:
        dataloader = [batch.to(device) for batch in dataloader]

    #! 创建模型, 并使用并行计算, 参数并行计算的模型的参数要保持一致, 也就是说, 在两张卡上同时进行的模型是一样的
    generator = Generator(args, subgraphs_num, nodes_num_list, node_attrs_dim)
    if args.cuda:
        generator.to(device)
        
    #! 传递 find_unused_parameters=True, 这样会帮助检测和处理未使用的参数
    DP_generator = DDP(generator, device_ids=[rank], find_unused_parameters=True)
    DP_generator.perturbation_matrices = generator.perturbation_matrices
    DP_generator.perturbation_biases = generator.perturbation_biases
    DP_generator.masking_matrices = generator.masking_matrices
    
    optimizer = torch.optim.Adam(
        params=DP_generator.parameters(),
        lr=args.generation_lr
    )

    Augmentaed_SubGraph_list = []

    #! 这里是生成 Hard Negative Samples, 其实就是生成 perturbation_matrices 等
    pbar = tqdm(range(1))
    for epoch in pbar:
        pbar.set_description('Hard Negative Samples Generation Epoch %d...' % epoch)
        
        #! 这里的每一个 batch 是 batch_size 个子图组成的节点
        batch_num = len(dataloader)
        for batch_id, batch in enumerate(dataloader):
            optimizer.zero_grad()

            adjs_dense, perturbation_adjs, masking_matrix, predicted_results, perturbation_predicted_results, masking_predicted_results, _, _ = DP_generator(
                batch, device)
            sim_loss = similarity_loss(adjs_dense, perturbation_adjs, masking_matrix)
            kl_loss = kl_div(predicted_results, perturbation_predicted_results, masking_predicted_results)
            l = sim_loss - kl_loss

            l.backward()
            optimizer.step()

            pbar.set_postfix(sim_loss=sim_loss.item(), kl_loss=kl_loss.item())
            Current_logger.info('Current_PID: {}; Epoch: {}/{}; Batch: {}/{}; sim_loss: {}; kl_loss: {}.'.format(current_pid, epoch+1, args.generation_epochs,
                                                                                                                batch_id+1, batch_num,
                                                                                                                    sim_loss.item(), kl_loss.item()))
    #! 这里则是生成 Augmented Graphs
    pbar = tqdm(range(len(current_subgraph_list)))
    pbar.set_description('Augmented Graphs Generation...')
    for i in pbar:
        each = current_subgraph_list[i]
        if args.cuda:
            each = each.to(device)

        Augmentaed_SubGraph = copy.deepcopy(each)

        p_matrix = DP_generator.perturbation_matrices[each.id]
        p_bias = DP_generator.perturbation_biases[each.id]
        m_matrix = DP_generator.masking_matrices[each.id]

        values = torch.Tensor([1 for i in range(each.edge_index.size()[1])])
        if args.cuda:
            values = values.to(device)
        adjs = torch.sparse_coo_tensor(each.edge_index, values, (each.num_nodes, each.num_nodes), dtype=torch.float)
        adjs_dense = adjs.to_dense()
        perturbation_adjs = torch.mm(p_matrix, adjs_dense) + p_bias
        perturbation_adjs = torch.sigmoid(perturbation_adjs)
        perturbation_adjs = torch.where(perturbation_adjs <= args.gamma, torch.zeros_like(perturbation_adjs),
                                        torch.ones_like(perturbation_adjs))
        perturbation_adjs_sparse = perturbation_adjs.to_sparse()
        Augmentaed_SubGraph.edge_index = perturbation_adjs_sparse.indices()

        masking_matrices = torch.sigmoid(m_matrix)
        masking_matrices = torch.where(masking_matrices <= args.gamma, torch.zeros_like(masking_matrices),
                                       torch.ones_like(masking_matrices))
        masked_attrs = torch.mul(masking_matrices, each.x)
        Augmentaed_SubGraph.x = masked_attrs
        
        Augmentaed_SubGraph_list.append(Augmentaed_SubGraph)
        Current_logger.info('Current_PID: {}; Augmented Subgraph {} generated'.format(current_pid, i+1))
            
    #! 这里会出现资源竞争问题
    #! 使用锁来避免资源竞争
    with lock:
        #! 记录子进程返回值
        print('Recieve Result from rank {}'.format(rank))
        Current_logger.info('Current_PID: {}; Recieve Result from rank {}'.format(current_pid, rank))
        return_queue.put((rank, 'woqu1'))  # 将进程的返回值放入队列
    dist.destroy_process_group()
