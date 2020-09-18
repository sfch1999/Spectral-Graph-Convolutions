import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

from Explain_Model import Explainer
from Model import GCN

with open('Pickles/preds.pickle', 'rb') as handle:
    labels = pickle.load(handle).astype(np.long)
with open('BestModel/train_validation.pickle', 'rb') as handle:
    cg_dict = pickle.load(handle)

train_idx = cg_dict['train']
test_idx = cg_dict['validation']
X = cg_dict['x']
adj = cg_dict['adj']

node_idx = 100

labels = torch.tensor(labels, dtype=torch.long)
labels_train = labels[train_idx]
labels_test = labels[test_idx]

model = GCN(nfeat=X.shape[1],
            nhid2=100,
            nhid1=50,
            nhid0=10,
            nclass=labels.max().item() + 1,
            dropout=0.5)
model.load_state_dict(torch.load('BestModel/model.pth.tar'))

explainer = Explainer(
    model=model,
    adj=adj,
    feat=X,
    label=labels,
    pred=model(X, adj),
    train_idx=train_idx,
    test_idx=test_idx,
    print_training=True,
    graph_mode=False,
)
mask,acc = explainer.explain_graph()

print(acc)

with open('BestModel/feat_mask.pickle', 'wb') as handle:
    pickle.dump(mask, handle, protocol=pickle.HIGHEST_PROTOCOL)

plt.imshow(torch.tensor(X) * torch.tensor(mask))
plt.colorbar()
plt.show()

# TODO: Excel sheet
