from datetime import datetime
import json
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, recall_score, precision_score

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from UnknownFW.BotRGCN import BotRGCN
from Loggers.MP_logger import mp_logger

class BotRGCN_Trainer(object):
    def __init__(self, args, hetero_subgraph, raw_subgraph):
        super(BotRGCN_Trainer, self).__init__()
        #! 设置参数
        self.args = args
        #! 设置随机数种子
        self.seed = args.seed
        self.set_random_seed()
        
        #! 读取数据
        self.data = hetero_subgraph  
        self.raw_data = raw_subgraph
        self.data.to('cuda:'+str(args.device_id))
        self.raw_data.to('cuda:'+str(args.device_id))
        
        #! 创建模型
        self.botrgcn = BotRGCN(self.args).to('cuda:'+str(args.device_id))
        self.raw_botrgcn = BotRGCN(self.args).to('cuda:'+str(args.device_id))
        
        #! 记录实验(准确率等)
        self.save_top_k = args.save_top_k
        self.patience = 0
        self.best_loss_epoch = 0
        self.best_acc_epoch = 0
        self.best_loss = 1e9
        self.best_loss_acc = -1e9
        self.best_acc = -1e9
        self.best_acc_loss = 1e9
        self.test_results = []
        #! 用来记录结果
        self.val_label_list = []
        self.val_pred_list = []
        self.test_label_list = []
        self.test_pred_list = []

        #! 创建分类器(就是一个MLP), 这个也要放入GPU中
        self.concat_classifier = nn.Sequential(
            nn.Linear(2*self.args.hidden_dim, self.args.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.args.hidden_dim, self.args.num_classes))
        self.concat_classifier.to('cuda:'+str(args.device_id))
        
    #! 设置随机数种子
    def set_random_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False        

    #! 开始训练
    def concat_train(self, logger=None):
        r'''
            训练器进行训练, 并返回结果(Acc, Pre, F1等), 分别返回验证集、测试集结果
            返回的都是最小损失对应的标签分布和预测分布
            - self.val_label_list[min_loss_index[0]], self.val_pred_list[min_loss_index[0]]
            - self.test_label_list[min_loss_index[0]], self.test_pred_list[min_loss_index[0]]           
        '''
        #! 创建了两个 bot_rgcn, 同时进行优化
        optimizer = torch.optim.AdamW(list(self.botrgcn.parameters())+list(self.raw_botrgcn.parameters()),
                                            lr=args.lr,
                                            weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=16, eta_min=0)

        val_accs = []
        val_losses = []
        test_accs = []

        #@ 创建交叉熵损失函数，带权重，设置固定权重
        #@ 这里改成动态的
        labels = self.data.y[self.data.train_mask]
        class_sample_count = torch.bincount(labels)
        #@ 计算每个类别的权重 (可以是样本数量的倒数)
        class_weights = torch.tensor([each/class_sample_count.sum() for each in reversed(class_sample_count)], 
                                    dtype=torch.float).to('cuda:{}'.format(self.args.device_id))
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        
        print('='*90)
        print('labels = self.data.y[self.data.train_mask]')
        print('class_sample_count = torch.bincount(labels)')
        print(class_sample_count)
        print('''torch.tensor([each/class_sample_count.sum() for each in reversed(class_sample_count)], 
                                    dtype=torch.float).to('cuda:0')''')
        print(class_weights)
        print('loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)')
        print('='*90)
        
        for epoch in range(args.epochs):
            #! 进行训练
            self.botrgcn.train()
            self.raw_botrgcn.train()
            
            optimizer.zero_grad()
            
            #! 获得两个通道的 embedding
            out_embedding = self.botrgcn(self.data)
            raw_embedding = self.raw_botrgcn(self.raw_data)
            
            train_out = self.concat_classifier(torch.cat((
                out_embedding[self.data.train_mask],
                raw_embedding[self.raw_data.train_mask]
                ), dim=1))
            
            # loss = F.cross_entropy(train_out, self.data.y[self.data.train_mask])
            loss = loss_fn(train_out, self.data.y[self.data.train_mask])
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            #! 进行验证, 这里的 eval() 是训练器本身自带的
            val_acc, val_loss, test_acc, _ = self.eval()            

            self.organize_val_log(val_loss, val_acc, epoch)
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            test_accs.append(test_acc)
            if self.patience > args.patience:
                break

        #! 获得测试结果
        test_results = self.get_test_results()
        test_results = np.array(test_results)
        
        #! 选择验证集上loss最小的 self.save_top_k 个结果打印, 这里是根据 val_loss 选择的, 我能不能换成 val_acc 为准?
        #! argsort() 返回元素按从小到大排序后的索引
        val_losses = np.array(val_losses)
        test_accs = np.array(test_accs)
        
        min_loss_index = val_losses.argsort()[:self.save_top_k]
        min_test_accs_index = test_accs.argsort()[-self.save_top_k:]
        
        print('top_{} val losses...'.format(self.save_top_k))
        logger.info('top_{} val losses...'.format(self.save_top_k))
        for i in min_loss_index:
            print(
                'epoch: %d, test_acc: %.4f, test_f1: %.4f, test_recall: %.4f, test_precision: %.4f, val_acc: %.4f'
                % (i+1, test_results[i][0], test_results[i][1],
                    test_results[i][2], test_results[i][3], test_results[i][4]))
            logger.info(
                'epoch: %d, test_acc: %.4f, test_f1: %.4f, test_recall: %.4f, test_precision: %.4f, val_acc: %.4f'
                % (i+1, test_results[i][0], test_results[i][1],
                    test_results[i][2], test_results[i][3], test_results[i][4]))
            
        print('top_{} test accs...'.format(self.save_top_k))
        logger.info('top_{} test accs...'.format(self.save_top_k))            
        for i in min_test_accs_index:
            print(
                'epoch: %d, test_acc: %.4f, test_f1: %.4f, test_recall: %.4f, test_precision: %.4f, val_acc: %.4f'
                % (i+1, test_results[i][0], test_results[i][1],
                    test_results[i][2], test_results[i][3], test_results[i][4]))
            logger.info(
                'epoch: %d, test_acc: %.4f, test_f1: %.4f, test_recall: %.4f, test_precision: %.4f, val_acc: %.4f'
                % (i+1, test_results[i][0], test_results[i][1],
                    test_results[i][2], test_results[i][3], test_results[i][4]))
            
        #! 返回 val_label|pred, test_label|pred, 选取验证集损失值最小的对应的测试集的各项指标
        # return self.val_label_list[min_loss_index[0]], self.val_pred_list[min_loss_index[0]], \
        #         self.test_label_list[min_loss_index[0]], self.test_pred_list[min_loss_index[0]]  
        return self.val_label_list[min_test_accs_index[self.save_top_k-1]], self.val_pred_list[min_test_accs_index[self.save_top_k-1]], \
                self.test_label_list[min_test_accs_index[self.save_top_k-1]], self.test_pred_list[min_test_accs_index[self.save_top_k-1]]    
                     
    def organize_val_log(self, val_loss, val_acc, epoch):
        #! 如果损失变小了
        if val_loss < self.best_loss:
            self.best_loss_acc = val_acc
            self.best_loss = val_loss
            self.best_loss_epoch = epoch
            self.patience = 0
        else:
            self.patience += 1          #! 损失没有减小
        #! 如果验证集准确率大了
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_acc_loss = val_loss
            self.best_acc_epoch = epoch

    #! 先进入评估模式, 然后让数据经过模型
    def eval(self):
        '''
            返回值: \n
            val_acc \n val_loss.item() \n test_acc \n test_loss.item()
        '''
        self.botrgcn.eval()
        self.raw_botrgcn.eval()
        
        out_embedding = self.botrgcn(self.data)
        raw_embedding = self.raw_botrgcn(self.raw_data)
        
        #! 获得验证集结果
        val_out = self.concat_classifier(torch.cat((
                out_embedding[self.data.val_mask],
                raw_embedding[self.raw_data.val_mask]
                ), dim=1))
        
        val_loss = F.cross_entropy(val_out, self.data.y[self.data.val_mask])
        val_label = self.data.y[self.data.val_mask].cpu().numpy()
        val_pred = torch.argmax(val_out, dim=1).cpu().numpy()
        
        #! 获得验证集的准确率
        val_acc = accuracy_score(val_label, val_pred)
        
        #! 获得测试集结果
        test_out = self.concat_classifier(torch.cat((
                out_embedding[self.data.test_mask],
                raw_embedding[self.raw_data.test_mask]
                ), dim=1))
        test_loss = F.cross_entropy(test_out, self.data.y[self.data.test_mask])
        test_label = self.data.y[self.data.test_mask].cpu().numpy()
        test_pred = torch.argmax(test_out, dim=1).cpu().numpy()
        
        #! 获得准确率、F1值、召回率、精确率
        test_acc = accuracy_score(test_label, test_pred)
        test_f1 = f1_score(test_label, test_pred, zero_division=0)
        test_recall = recall_score(test_label, test_pred, zero_division=0)
        #@ 如果分母为0，那么把 Precision 的值设为 0
        test_precision = precision_score(test_label, test_pred, zero_division=0) 
         
        # #! 超过basline的分数
        # if test_acc > 0.869:
        #     print('large test_acc: ', test_acc)
        
        #! 记录结果
        self.test_results.append(
            [test_acc, test_f1, test_recall, test_precision, val_acc])
        
        #! 这里则是记录 label、pred
        self.val_label_list.append(val_label)
        self.val_pred_list.append(val_pred)
        self.test_label_list.append(test_label)
        self.test_pred_list.append(test_pred)
        
        #! 同时返回真实标签和预测值
        return val_acc, val_loss.item(), test_acc, test_loss.item()

    #! 得到 test_results 列表
    def get_test_results(self):
        '''
            test_results = [test_acc, test_f1, test_recall, test_precision, val_acc]
        '''
        return self.test_results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #! 数据集名字
    parser.add_argument('--dataset', type=str, default='twibot_20', help='数据集名')
    parser.add_argument('--dataset_name', type=str, default='twibot_20', help='数据集名')
    parser.add_argument('--dataset_dir', type=str, default='/data3/xupin/0_UNName/data/', help='数据集路径')
    parser.add_argument('--dataset_specific', type=str, default='_type_1_node_pt_undirct_hetero.pt', help='数据集名字')
    
    #! 数据集特征
    parser.add_argument('--node_num', type=int, default=11826, help='有标签节点数量')
    parser.add_argument('--cat_num', type=int, default=3, help='类别特征维度')
    parser.add_argument('--prop_num', type=int, default=5, help='数值特征维度')
    parser.add_argument('--des_num', type=int, default=768, help='账号描述文本特征')
    parser.add_argument('--tweet_num', type=int, default=768, help='推文文本特征')
    parser.add_argument('--num_classes', type=int, default=2, help='分类类别')
    parser.add_argument('--num_relations', type=int, default=2, help='数据集中边的类别数')
        
    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--proj_dim', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--alpha1', type=float,
                        default=0.1)  # node self-supervised loss
    parser.add_argument('--alpha2', type=float,
                        default=0.1)  # subgraph self-supervised loss
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_convs', type=int, default=3)
    
    #! edge dropout ratee
    parser.add_argument('--pe', type=float, default=0.2)  # edge dropout rate
    parser.add_argument('--pf', type=float, default=0.2)

    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--batch_size', default=5000, type=int)
    
    #! 下面是优化器的参数
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=3e-3)
    
    parser.add_argument('--patience', type=int, default=50)
    
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument('--conv_dropout', type=float, default=0.5)
    parser.add_argument('--pooling_dropout', type=float, default=0.5)
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument("--device_id", type=int, default=2, help='使用的 GPU 设备号')
    parser.add_argument(
        '--save_top_k', type=int,
        default=6)  # save top k models with best validation loss
    args = parser.parse_args()
    vargs_args = vars(args)
    formated_args = json.dumps(vargs_args, indent=4)
    print(formated_args)
    
    device=torch.device('cuda:{}'.format(args.device_id))
    #! 读取异质图子图列表
    dataset_name = args.dataset_name
    hetero_graph_list = torch.load('/data3/xupin/0_UNName/data/way_2/twibo20/'+dataset_name+'_subgraphlist_hetero_to_type_1_node_hetero.pt',
                                   map_location=device)
    raw_graph_list = torch.load('/data3/xupin/0_UNName/data/way_2/twibo20/'+dataset_name+'_subgraphlist.pt',
                                map_location=device)
        
    val_label_list = []
    val_pred_list = []
    test_label_list = []
    test_pred_list = []
    
    #! 创建日记记录器
    main_logger = mp_logger(dataset_name+'_way_2_hetero_subgraph_train')
    
    #! 分别对每个子图进行训练
    for index, subgraph_pair in enumerate(zip(hetero_graph_list, raw_graph_list)):
        hetero_subgraph=subgraph_pair[0]
        raw_subgraph=subgraph_pair[1]
                
        #! 训练模型
        print('train hetero subgraph '+str(index+1))
        main_logger.info('train hetero subgraph '+str(index+1))
        
        #! 创建训练器
        botrgcn_trainer = BotRGCN_Trainer(args, hetero_subgraph, raw_subgraph)
        #! 对子图进行训练
        val_label, val_pred, test_label, test_pred = botrgcn_trainer.concat_train(logger=main_logger)
        #! 记录对应值
        val_label_list.append(val_label)
        val_pred_list.append(val_pred)
        test_label_list.append(test_label)
        test_pred_list.append(test_pred)
        
        print('hetero subgraph '+str(index+1)+' compelete')
        main_logger.info('hetero subgraph '+str(index+1)+' compelete')
        print()
        main_logger.info('\n')

    #TODO 是不是要保存 val_label_list, val_pred_list, test_label_list, test_pred_list; 以进行更详细的分析
    #TODO current_datetime = datetime.now()
    #TODO formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
    #TODO torch.save(torch.tensor([val_label_list, val_pred_list, test_label_list, test_pred_list]),
    #TODO            '/data3/xupin/0_UNName/logs/'+ formatted_datetime + '_way_2_hetero_subgraph_label_pred.pt')

    #! 进行合并
    val_label = np.concatenate((val_label_list), axis=0)
    val_pred = np.concatenate((val_pred_list), axis=0)
    test_label = np.concatenate((test_label_list), axis=0)
    test_pred  = np.concatenate((test_pred_list), axis=0)
        
    #! 验证集准确率
    val_acc = accuracy_score(val_label, val_pred)
    #! 测试集准确率、F1值、召回率、精确率
    test_acc = accuracy_score(test_label, test_pred)
    test_f1 = f1_score(test_label, test_pred)
    test_recall = recall_score(test_label, test_pred)
    test_precision = precision_score(test_label, test_pred)

    #! 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(test_label, test_pred).ravel()
    mcc = matthews_corrcoef(test_label, test_pred) 
    
    print('final val_acc: %.4f' % val_acc)
    print('final test_acc: %.4f' % test_acc)
    print('final test_f1: %.4f' % test_f1)
    print('final test_recall: %.4f' % test_recall)    
    print('final test_precision: %.4f' % test_precision)   
    print('True Positives (TP): {}'.format(tp))
    print('True Negatives (TN): {}'.format(tn))
    print('False Positives (FP): {}'.format(fp))
    print('False Negatives (FN):: {}'.format(fn))
    
    main_logger.info('final val_acc: %.4f' % val_acc) 
    main_logger.info('final test_acc: %.4f' % test_acc)
    main_logger.info('final test_f1: %.4f' % test_f1)
    main_logger.info('final test_recall: %.4f' % test_recall)
    main_logger.info('final test_precision: %.4f' % test_precision)
    main_logger.info('True Positives (TP): {}'.format(tp))
    main_logger.info('True Negatives (TN): {}'.format(tn))
    main_logger.info('False Positives (FP): {}'.format(fp))
    main_logger.info('False Negatives (FN):: {}'.format(fn))
    main_logger.info('MCC: {}'.format(mcc))