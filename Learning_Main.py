import pickle
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from Model import pairwise_distances, GCN, accuracy
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from util import load_data,load_data2
from torch.autograd import Variable


# with open('Pickles/feats.pickle', 'rb') as handle:
#     feats = pickle.load(handle)
# with open('Pickles/age_adj.pickle', 'rb') as handle:
#     age_adj = pickle.load(handle)
# with open('Pickles/preds.pickle', 'rb') as handle:
#     labels = pickle.load(handle).astype(np.long)
#
# # print(feats)
# # print(np.mean(feats,axis=0))
# # print()
# # exit()
# adj = (1 / pairwise_distances(torch.tensor(feats / np.expand_dims(np.mean(feats, axis=0), axis=0))))
# max_elemnt = torch.max(adj[torch.where(adj < 1e+6)])
# for i in range(feats.shape[0]):
#     adj[i, i] = max_elemnt * 2
# d_hat_inv = np.linalg.inv(np.diag(torch.sum(adj, dim=1))) ** (1 / 2)
# temp = np.matmul(d_hat_inv, adj)
# adj = np.matmul(temp, d_hat_inv)
# adj = torch.tensor(adj, dtype=torch.float)

feats, adj_mat, labels, train_idx, test_idx, labels_train, labels_test = load_data(True)
x, adj, labels = torch.tensor(feats,dtype=torch.float), Variable(adj_mat), Variable(torch.tensor(labels,dtype=torch.long))


model = GCN(nfeat=feats.shape[1],
            nhid2=100,
            nhid1=50,
            nhid0=10,
            nclass=labels.max().item() + 1,
            dropout=0.5)
# model.double()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.000001)
# optimizer = optim.Adam(model.parameters(),
#                        lr=1, weight_decay=5e-4)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# num_nodes = labels.shape[0]
# num_train = int(num_nodes * 0.8)
# idx = [i for i in range(num_nodes)]
# np.random.shuffle(idx)
# train_idx = idx[:num_train]
# test_idx = idx[num_train:]
# labels_train = torch.tensor(labels[train_idx], dtype=torch.long)
# labels_test = torch.tensor(labels[test_idx], dtype=torch.long)
# x = torch.tensor(feats, dtype=torch.float, requires_grad=True)
lr_scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=50, verbose=True)
# criterion = nn.CrossEntropyLoss()


# print(torch.sum(adj, dim=1))
# exit()


# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(pytorch_total_params)
# exit()

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(x, adj)
    # if epoch<400:
    loss_train = F.nll_loss(output[train_idx], labels_train)
    # if epoch % 20 == 0:
    #     print(output)
    # loss_train = F.nll_loss(output, labels_train)-(torch.sum(torch.abs(output[0]-output[1]))+torch.sum(torch.abs(output[1]-output[2]))+torch.sum(torch.abs(output[2]-output[3])))
    acc_train = accuracy(output[train_idx], labels_train)
    loss_train.backward()
    optimizer.step()

    # Evaluate validation set performance separately,
    # deactivates dropout during validation run.
    loss_val = F.nll_loss(output[test_idx], labels_test)
    acc_val = accuracy(output[test_idx], labels_test)
    lr_scheduler.step(loss_train)
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return acc_val.item()


# Train model
t_total = time.time()
best_val_acc = 0
best_model_state_dict = None

for epoch in range(300):
    acc_val = train(epoch)
    if acc_val > best_val_acc:
        best_val_acc = acc_val
        best_model_state_dict = model.state_dict()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print("Best validation accuracy: {:.4f}".format(best_val_acc))

cg = {'train': train_idx,
      'validation': test_idx,
      'x': x,
      'adj': adj
      }

# with open('BestModel/train_validation.pickle', 'wb') as handle:
#     pickle.dump(cg, handle, protocol=pickle.HIGHEST_PROTOCOL)
# torch.save(best_model_state_dict, 'BestModel/model.pth.tar')
