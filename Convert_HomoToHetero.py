import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.data import Batch

from UnknownFW.load_data import merge_augmented_subgraphlist
from UnknownFW.load_data import load_pt_raw_data
from UnknownFW.load_data import homo_to_hetero
from UnknownFW.Graph_Split import Cluster_Split

if __name__ == '__main__':
    type_list = ['type_1','type_2']
    # merge_augmented_subgraphlist('/data3/xupin/0_UNName/data/', '/data3/xupin/0_UNName/processed_data/', type_list)
    homo_to_hetero('/data3/xupin/0_UNName/data/', type_list)