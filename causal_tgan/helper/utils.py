import os
import csv
import pickle
import networkx as nx
import numpy as np
from causal_tgan.dataset import DataTransformer, GeneralTransformer, PlainTransformer

def adjacency_to_parent_list(adj, names):
    """
    Convert adjacency matrix (adj[i, j] = 1 means names[i] -> names[j])
    to parent list format: [[child, [parent1, parent2, ...]], ...]
    """
    adj = np.asarray(adj)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be a square matrix")
    n = adj.shape[0]
    if len(names) != n:
        raise ValueError("names length must match adj size")

    # binarize just in case (accepts 0/1 or real weights)
    adj = (adj != 0).astype(int)

    parent_list = []
    for j in range(n):
        parents = [names[i] for i in range(n) if adj[i, j] == 1]
        parent_list.append([names[j], parents])
    return parent_list

def get_transformer(transformer_type):
    if transformer_type == 'general':
        return GeneralTransformer()
    elif transformer_type == 'plain':
        return PlainTransformer()
    elif transformer_type == 'ctgan':
        return DataTransformer()
    else:
        raise ('Transformer type of {} does not exist, should be one of [\'general\', \'tablegan\', \'ctgan\']'.format(transformer_type))

def data_transform(transformer_type, data, discrete_cols):
    transformer = get_transformer(transformer_type)
    transformer.fit(data, discrete_cols)
    transform_data, data_dims = transformer.transform(data)

    return transform_data, transformer, data_dims

def load_options(options_file_name):
    """ Loads the training, model, and noise configurations from the given folder """
    with open(os.path.join(options_file_name), 'rb') as f:
        train_options = pickle.load(f)
        gan_config = pickle.load(f)

    return train_options, gan_config

def print_progress(losses_accu):
    max_len = max([len(loss_name) for loss_name in losses_accu])
    for loss_name, loss_value in losses_accu.items():
        print(loss_name.ljust(max_len+4) + '{:.4f}'.format(np.mean(loss_value)))

def write_losses(file_name, losses_accu, epoch, duration):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()] + ['duration']
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(np.mean(loss_list)) for loss_list in losses_accu.values()] + ['{:.0f}'.format(duration)]
        writer.writerow(row_to_write)

def restore_feature_info(folder_path):
    t_path = os.path.join(folder_path, 'transformer.pickle')
    f_path = os.path.join(folder_path, 'featureInfo.pickle')
    graph_path = os.path.join(folder_path, 'causal_graph.pickle')
    with open(t_path, 'rb') as f:
        transformer = pickle.load(f)
    with open(f_path, 'rb') as f:
        feature_info = pickle.load(f)
    with open(graph_path, 'rb') as f:
        causal_graph = pickle.load(f)

    return transformer, feature_info, causal_graph

def _adjMatrix2graph(adjMatrix, col_names):
    graph = [[item, []] for item in col_names]
    for idx, c_nodes in enumerate(adjMatrix):
        c_idx = np.where(np.asarray(c_nodes)==1)
        for i in c_idx[0]:
            graph[i][1].append(col_names[idx])

    return graph

def topology_order(amat):
    order = []
    amat = amat.copy()
    num_node = len(amat[0])
    while True:
        tmp = amat.sum(axis=0)
        cur_root = [i for i in range(num_node) if (tmp[i]==0) & (i not in order)]
        for idx in cur_root:
            amat[idx] = [0 for _ in range(num_node)]
        order.extend(cur_root)
        if len(cur_root) == 0:
            break
    return order

def _no_cycle(amat):
    G = nx.from_numpy_matrix(amat, create_using=nx.DiGraph)
    try:
        tmp = next(nx.simple_cycles(G))
        return False
    except:
        return True

def read_names(path):
    col_names = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.strip()
            col_names.append(tmp.replace("\"" ,""))
    return col_names

def read_amat(path):
    adj_matrix = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.split(' ')
            tmp = [int(item) for item in tmp]
            adj_matrix.append(tmp)
    return np.asarray(adj_matrix)

