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

from UnknownFW.BotRGCN_mgtab import BotRGCN
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

        self.botrgcn = BotRGCN(self.args).to('cuda:'+str(args.device_id))
        self.raw_botrgcn = BotRGCN(self.args).to('cuda:'+str(args.device_id))
        
        self.save_top_k = args.save_top_k
        self.patience = 0
        self.best_loss_epoch = 0
        self.best_acc_epoch = 0
        self.best_loss = 1e9
        self.best_loss_acc = -1e9
        self.best_acc = -1e9
        self.best_acc_loss = 1e9
        self.test_results = []

        self.val_label_list = []
        self.val_pred_list = []
        self.test_label_list = []
        self.test_pred_list = []

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

    def concat_train(self, logger=None):
        optimizer = torch.optim.AdamW(list(self.botrgcn.parameters())+list(self.raw_botrgcn.parameters()),
                                            lr=args.lr,
                                            weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=16, eta_min=0)

        val_accs = []
        val_losses = []
        test_accs = []
        
        labels = self.data.y[self.data.train_mask]
        class_sample_count = torch.bincount(labels)
        class_weights = torch.tensor([each/class_sample_count.sum() for each in reversed(class_sample_count)], 
                                    dtype=torch.float).to('cuda:{}'.format(self.args.device_id))
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        
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

            val_acc, val_loss, test_acc, _ = self.eval()            

            self.organize_val_log(val_loss, val_acc, epoch)
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            test_accs.append(test_acc)
            if self.patience > args.patience:
                break

        test_results = self.get_test_results()
        test_results = np.array(test_results)
        
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
            
        return self.val_label_list[min_test_accs_index[self.save_top_k-1]], self.val_pred_list[min_test_accs_index[self.save_top_k-1]], \
                self.test_label_list[min_test_accs_index[self.save_top_k-1]], self.test_pred_list[min_test_accs_index[self.save_top_k-1]]    
                     
    def organize_val_log(self, val_loss, val_acc, epoch):
        if val_loss < self.best_loss:
            self.best_loss_acc = val_acc
            self.best_loss = val_loss
            self.best_loss_epoch = epoch
            self.patience = 0
        else:
            self.patience += 1
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_acc_loss = val_loss
            self.best_acc_epoch = epoch

    def eval(self):
        '''
            返回值: \n
            val_acc \n val_loss.item() \n test_acc \n test_loss.item()
        '''
        self.botrgcn.eval()
        self.raw_botrgcn.eval()
        out_embedding = self.botrgcn(self.data)
        raw_embedding = self.raw_botrgcn(self.raw_data)
        
        val_out = self.concat_classifier(torch.cat((
                out_embedding[self.data.val_mask],
                raw_embedding[self.raw_data.val_mask]
        ), dim=1))
        
        val_loss = F.cross_entropy(val_out, self.data.y[self.data.val_mask])
        val_label = self.data.y[self.data.val_mask].cpu().numpy()
        val_pred = torch.argmax(val_out, dim=1).cpu().numpy()
        
        val_acc = accuracy_score(val_label, val_pred)
        
        test_out = self.concat_classifier(torch.cat((
                out_embedding[self.data.test_mask],
                raw_embedding[self.raw_data.test_mask]
                ), dim=1))
        test_loss = F.cross_entropy(test_out, self.data.y[self.data.test_mask])
        test_label = self.data.y[self.data.test_mask].cpu().numpy()
        test_pred = torch.argmax(test_out, dim=1).cpu().numpy()
        
        test_acc = accuracy_score(test_label, test_pred)
        test_f1 = f1_score(test_label, test_pred, zero_division=0)
        test_recall = recall_score(test_label, test_pred, zero_division=0)
        test_precision = precision_score(test_label, test_pred, zero_division=0) 
         
        self.test_results.append(
            [test_acc, test_f1, test_recall, test_precision, val_acc])
        
        self.val_label_list.append(val_label)
        self.val_pred_list.append(val_pred)
        self.test_label_list.append(test_label)
        self.test_pred_list.append(test_pred) 
        return val_acc, val_loss.item(), test_acc, test_loss.item()

    def get_test_results(self):
        '''
            test_results = [test_acc, test_f1, test_recall, test_precision, val_acc]
        '''
        return self.test_results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #! 数据集名字
    parser.add_argument('--dataset', type=str, default='mgtab', help='数据集名')
    parser.add_argument('--dataset_name', type=str, default='mgtab', help='数据集名')
    parser.add_argument('--dataset_dir', type=str, default='/data3/data/', help='数据集路径')
    parser.add_argument('--dataset_specific', type=str, default='_type_1_node_pt_undirct_hetero.pt', help='数据集名字')
    
    #! 数据集特征
    parser.add_argument('--node_num', type=int, default=11826, help='有标签节点数量')
    parser.add_argument('--cat_num', type=int, default=3, help='类别特征维度')
    parser.add_argument('--prop_num', type=int, default=5, help='数值特征维度')
    parser.add_argument('--des_num', type=int, default=768, help='账号描述文本特征')
    parser.add_argument('--tweet_num', type=int, default=768, help='推文文本特征')
    parser.add_argument('--num_classes', type=int, default=2, help='分类类别')
    parser.add_argument('--num_relations', type=int, default=7, help='数据集中边的类别数')
    
    #! 记得修改输入特征维度
    parser.add_argument('--input_dim', type=int, default=788)
    parser.add_argument('--hidden_dim', type=int, default=32)
    
    #! 再过一遍 GCL(图对比学习?)
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--batch_size', default=5000, type=int)
    
    #! 下面是优化器的参数
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay for optimizer')
    
    parser.add_argument('--patience', type=int, default=50)
    
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument('--epochs', default=200, type=int)
    # parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument("--device_id", type=int, default=2, help='使用的 GPU 设备号')
    parser.add_argument(
        '--save_top_k', type=int,
        default=6)  # save top k models with best validation loss
    args = parser.parse_args()
    vargs_args = vars(args)
    formated_args = json.dumps(vargs_args, indent=4)
    print(formated_args)
    
    dataset_name = args.dataset_name
    hetero_graph_list = torch.load('/data3/data/way_2/'+dataset_name+'_subgraphlist_augmented.pt',
                                   map_location='cuda:{}'.format(args.device_id))
    raw_graph_list = torch.load('/data3/data/way_2/'+dataset_name+'_subgraphlist.pt',
                                map_location='cuda:{}'.format(args.device_id))
        
    val_label_list = []
    val_pred_list = []
    test_label_list = []
    test_pred_list = []
    
    main_logger = mp_logger(dataset_name+'_way_2_hetero_subgraph_train')
    
    for index, subgraph_pair in enumerate(zip(hetero_graph_list, raw_graph_list)):
        hetero_subgraph=subgraph_pair[0]
        raw_subgraph=subgraph_pair[1]
                
        print('train hetero subgraph '+str(index+1))
        main_logger.info('train hetero subgraph '+str(index+1))
        
        botrgcn_trainer = BotRGCN_Trainer(args, hetero_subgraph, raw_subgraph)
        val_label, val_pred, test_label, test_pred = botrgcn_trainer.concat_train(logger=main_logger)
        val_label_list.append(val_label)
        val_pred_list.append(val_pred)
        test_label_list.append(test_label)
        test_pred_list.append(test_pred)
        
        print('hetero subgraph '+str(index+1)+' compelete')
        main_logger.info('hetero subgraph '+str(index+1)+' compelete')
        print()
        main_logger.info('\n')


    val_label = np.concatenate((val_label_list), axis=0)
    val_pred = np.concatenate((val_pred_list), axis=0)
    test_label = np.concatenate((test_label_list), axis=0)
    test_pred  = np.concatenate((test_pred_list), axis=0)
        
    val_acc = accuracy_score(val_label, val_pred)
    test_acc = accuracy_score(test_label, test_pred)
    test_f1 = f1_score(test_label, test_pred)
    test_recall = recall_score(test_label, test_pred)
    test_precision = precision_score(test_label, test_pred)

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
