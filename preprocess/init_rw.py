import numpy as np
import json
import time
from pathlib import Path

import tqdm
from torch_geometric.utils import from_scipy_sparse_matrix
import torch
import compute_merw as rw
import scipy
import argparse
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')
DATA_PATH = ".."

if __name__ == '__main__':
        # for data_name in ["ICEWS14", "ICEWS18", "ICEWS05-15", "GDELT"]:
        for data_name in ["ICEWS14"]:
                n = np.load(f"{DATA_PATH}/path_data/{data_name}/y.npy")
                edge_index = np.load(f"{DATA_PATH}/path_data/{data_name}/edge_index.npy")
                row = edge_index[0]
                col = edge_index[1]
                data = np.ones(edge_index.shape[-1])
                adj = csr_matrix((data, (row, col)),shape=(n, n))
                adj = adj + scipy.sparse.eye(n)  # with self-loop or not
                start = time.time()
                start_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start))
                print("calculating", start_time)

                #  MERW
                P_merw, _, _, _ = rw.compute_merw(adj)
                M = edge_index.shape[1]
                cal_end = time.time()
                print("saving", (cal_end-start)/60, (cal_end-start)/3600)
                # file = open(f"{DATA_PATH}/edge_input/{data_name}/{data_name}.in", "w")
                file = open(f"{DATA_PATH}/edge_input/{data_name}/{data_name}.in", "w")
                # y.shape[0] node,edge_index.shape[1]*2
                print(n, edge_index.shape[1]*2, file=file)
                for i in tqdm.tqdm(range(M)):
                    # u, v = edge_index[0, i], edge_index[1, i]
                    # print(u, v, P_merw[u, v], file=file)
                    # print(v, u, P_merw[v, u], file=file)
                    u, v, t = edge_index[0, i], edge_index[1, i], edge_index[2, i]
                    print(u, v, t, P_merw[u, v], file=file)
                    # print(v, u, t, P_merw[v, u], file=file)


                end = time.time()
                end_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(end))
                print("over", (end-start)/60, (end-start)/3600, end_time)
