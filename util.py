import pickle
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch


def build_optimizer(params, weight_decay=0.0):
    filter_fn = filter(lambda p: p.requires_grad, params)
    lr = 1
    opt = "adam"
    opt_scheduler = "none"
    opt_decay_step = 50
    if opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=lr, weight_decay=weight_decay)
    elif opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=lr, momentum=0.95, weight_decay=weight_decay)
    elif opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=lr, weight_decay=weight_decay)
    elif opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=lr, weight_decay=weight_decay)
    if opt_scheduler == 'none':
        return None, optimizer
    else:
        lr_scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.8, patience=50, verbose=True)

    return lr_scheduler, optimizer


def pairwise_distances(x, y=None):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return torch.clamp(dist, 0.0, np.inf)


def load_data(age: bool):
    with open('Pickles/feats.pickle', 'rb') as handle:
        feats = pickle.load(handle)
    with open('Pickles/age_adj.pickle', 'rb') as handle:
        age_adj = pickle.load(handle)
    with open('Pickles/preds.pickle', 'rb') as handle:
        labels = pickle.load(handle).astype(np.long)

    # norm_feats = (feats - np.mean(feats, axis=0)) / np.std(feats, axis=0)
    norm_feats = (feats - np.min(feats, axis=0)) / (np.max(feats, axis=0) - np.min(feats, axis=0))

    adj = pairwise_distances(torch.tensor(norm_feats))

    # print(adj)
    # exit()

    if age:
        for i in range(feats.shape[0]):
            adj[i, i] = 0
        adj = adj * age_adj

    max_element = torch.max(adj[torch.where(adj < 1e+6)])
    for i in range(feats.shape[0]):
        adj[i, i] = max_element * 2

    d_hat_inv = np.linalg.inv(np.diag(torch.sum(adj, dim=1))) ** (1 / 2)
    temp = np.matmul(d_hat_inv, adj)
    adj = np.matmul(temp, d_hat_inv)
    adj = torch.tensor(adj, dtype=torch.float)

    adj = torch.eye(labels.shape[0]) - adj

    num_nodes = labels.shape[0]
    num_train = int(num_nodes * 0.9)
    # num_train = 100
    idx = [i for i in range(num_nodes)]
    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]
    labels_train = torch.tensor(labels[train_idx], dtype=torch.long)
    labels_test = torch.tensor(labels[test_idx], dtype=torch.long)
    # adj = torch.eye(labels.shape[0])
    # adj = torch.tensor(age_adj)
    return norm_feats, adj, labels, train_idx, test_idx, labels_train, labels_test


def load_data2(age: bool):
    with open('Pickles/feats.pickle', 'rb') as handle:
        feats = pickle.load(handle)
    with open('Pickles/age_adj.pickle', 'rb') as handle:
        age_adj = pickle.load(handle)
    with open('Pickles/preds.pickle', 'rb') as handle:
        labels = pickle.load(handle).astype(np.long)

    adj = (1 / pairwise_distances(torch.tensor(feats / np.expand_dims(np.mean(feats, axis=0), axis=0))))

    if age:
        for i in range(feats.shape[0]):
            adj[i, i] = 0
        adj = adj * age_adj

    max_element = torch.max(adj[torch.where(adj < 1e+6)])
    for i in range(feats.shape[0]):
        adj[i, i] = max_element * 2
    d_hat_inv = np.linalg.inv(np.diag(torch.sum(adj, dim=1))) ** (1 / 2)
    temp = np.matmul(d_hat_inv, adj)
    adj = np.matmul(temp, d_hat_inv)
    adj = torch.tensor(adj, dtype=torch.float)

    num_nodes = labels.shape[0]
    num_train = int(num_nodes * 0.9)
    # num_train = 100
    idx = [i for i in range(num_nodes)]
    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]
    labels_train = torch.tensor(labels[train_idx], dtype=torch.long)
    labels_test = torch.tensor(labels[test_idx], dtype=torch.long)
    # adj = torch.eye(labels.shape[0])
    # adj = torch.tensor(age_adj)

    return feats, adj, labels, train_idx, test_idx, labels_train, labels_test
