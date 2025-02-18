import os
import copy
import random
import numpy as np
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

#! 两个损失
def similarity_loss(adjs_dense, perturbation_adjs, masking_matrix):
    diff = adjs_dense-perturbation_adjs
    diff_norm = torch.linalg.matrix_norm(diff, ord=1)/torch.ones_like(diff).sum()
    masking_matrix_norm = torch.linalg.matrix_norm(masking_matrix, ord=1)/torch.ones_like(masking_matrix).sum()
    return diff_norm-masking_matrix_norm

#! 进行 kl_loss 的计算
def kl_div(predicted_results, perturbation_predicted_results, masking_predicted_results):
    predicted_results = predicted_results.softmax(dim=1)
    perturbation_predicted_results = perturbation_predicted_results.log_softmax(dim=1)
    masking_predicted_results = masking_predicted_results.log_softmax(dim=1)
    loss_func = torch.nn.KLDivLoss(reduction='batchmean')
    kl_loss_1 = loss_func(perturbation_predicted_results, predicted_results)
    kl_loss_2 = loss_func(masking_predicted_results, predicted_results)
    return kl_loss_1+kl_loss_2


def generation(args, subgraphs_num, nodes_num_list, 
                    node_attrs_dim, subgraph_list, 
                        device, main_logger=None, mp_data_name=None, mp_queue=None):
    '''
        @Param: args | 参数列表 \n
        @Param: subgraphs_num | 子图数量 \n
        @Param: nodes_num_list | 子图节点数量列表 \n
        @Param: node_attrs_dim | 节点特征维度 \n
        @Param: subgraph_list | 子图列表 \n
        @Param: device | 使用的GPU设备号 \n
        @Param: main_logger | 日志记录器 \n
        @Param: mp_data_name | 该进程处理的数据名称 \n
        @Param: mp_queue | 保存多进程中每个进程的结果
    '''    
    #! 创建设备
    device = device
    
    #! 为每个进程创建 dataloader
    dataloader = DataLoader(subgraph_list, batch_size=args.batch_size, shuffle=True)
    dataloader = [batch for batch in dataloader]
    if args.cuda:
        dataloader = [batch.to(device) for batch in dataloader]

    #! 创建模型
    generator = Generator(args, subgraphs_num, nodes_num_list, node_attrs_dim)
    if args.cuda:
        generator.to(device)

    optimizer = torch.optim.Adam(
        params=generator.parameters(),
        lr=args.generation_lr
    )

    Augmentaed_SubGraph_list = []

    #! 这里是生成 Hard Negative Samples
    pbar = tqdm(range(args.generation_epochs))
    for epoch in pbar:
        pbar.set_description('Hard Negative Samples Generation Epoch %d...' % (epoch+1))
        
        #! 这里的每一个 batch 是 batch_size 个子图组成的节点
        batch_num = len(dataloader)
        for batch_id, batch in enumerate(dataloader):
            optimizer.zero_grad()

            adjs_dense, perturbation_adjs, masking_matrix, predicted_results, perturbation_predicted_results, masking_predicted_results, _, _ = generator(
                batch, device)
            #! 这里的 loss 都是针对和 Original 对比而言
            sim_loss = similarity_loss(adjs_dense, perturbation_adjs, masking_matrix)
            kl_loss = kl_div(predicted_results, perturbation_predicted_results, masking_predicted_results)
            l = sim_loss - kl_loss

            l.backward()
            optimizer.step()

            pbar.set_postfix(sim_loss=sim_loss.item(), kl_loss=kl_loss.item())
            main_logger.info('Epoch: {}/{}; Batch: {}/{}; sim_loss: {}; kl_loss: {}.'.format(epoch+1, args.generation_epochs,
                                                                                                batch_id+1, batch_num,
                                                                                                        sim_loss.item(), kl_loss.item()))    

    #! 这里则是生成 Augmented Graphs
    pbar = tqdm(range(len(subgraph_list)))
    pbar.set_description('Augmented Graphs Generation...')
    for i in pbar:
        # each = current_subgraph_list[i]
        each = subgraph_list[i]
        if args.cuda:
            each = each.to(device)


        Augmentaed_SubGraph = copy.deepcopy(each)

        p_matrix = generator.perturbation_matrices[each.id]
        p_bias = generator.perturbation_biases[each.id]
        m_matrix = generator.masking_matrices[each.id]

        values = torch.Tensor([1 for i in range(each.edge_index.size()[1])])
        if args.cuda:
            values = values.to(device)
        adjs = torch.sparse_coo_tensor(each.edge_index, values, (each.num_nodes, each.num_nodes), dtype=torch.float)
        adjs_dense = adjs.to_dense()
        #! 再次进行过滤
        perturbation_adjs = torch.mm(p_matrix, adjs_dense) + p_bias
        perturbation_adjs = torch.sigmoid(perturbation_adjs)
        perturbation_adjs = torch.where(perturbation_adjs <= args.gamma, torch.zeros_like(perturbation_adjs),
                                        torch.ones_like(perturbation_adjs))
        perturbation_adjs_sparse = perturbation_adjs.to_sparse()
        Augmentaed_SubGraph.edge_index = perturbation_adjs_sparse.indices()
        #! 再次进行过滤
        masking_matrices = torch.sigmoid(m_matrix)
        masking_matrices = torch.where(masking_matrices <= args.gamma, torch.zeros_like(masking_matrices),
                                       torch.ones_like(masking_matrices))
        masked_attrs = torch.mul(masking_matrices, each.x)
        Augmentaed_SubGraph.x = masked_attrs
        
        Augmentaed_SubGraph_list.append(Augmentaed_SubGraph)
        
        #! Record
        main_logger.info('Augmented Subgraph {} generated'.format(i+1))
 
    return Augmentaed_SubGraph_list
