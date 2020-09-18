import math
import os
import time
import torch.nn.functional as F

import torch
import numpy as np
import torch.nn as nn
import util
from Model import accuracy


def neighborhoods(adj, n_hops, use_cuda):
    """Returns the n_hops degree adjacency matrix adj."""
    if use_cuda:
        adj = adj.cuda()
    hop_adj = power_adj = adj
    for i in range(n_hops - 1):
        power_adj = power_adj @ adj
        prev_hop_adj = hop_adj
        hop_adj = hop_adj + power_adj
        hop_adj = (hop_adj > 0).float()
    return hop_adj.cpu().numpy().astype(int)


class Explainer:
    def __init__(
            self,
            model,
            adj,
            feat,
            label,
            pred,
            train_idx,
            test_idx,
            print_training=True,
            graph_mode=False,
    ):
        self.model = model
        self.model.eval()
        self.adj = adj
        self.feat = feat
        self.label = label
        self.pred = pred
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.n_hops = 2
        self.graph_mode = graph_mode
        self.neighborhoods = None if self.graph_mode else neighborhoods(adj=self.adj, n_hops=self.n_hops,
                                                                        use_cuda=False)
        self.print_training = print_training
        self.num_epochs = 100

        # Main method

    def explain_node(
            self, node_idx, graph_idx=0, model="exp"
    ):
        """Explain a single node prediction
        """
        # index of the query node in the new adj
        print("node label: ", self.label[node_idx])
        node_idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood(
            node_idx, graph_idx)
        print("neigh graph idx: ", node_idx, node_idx_new)

        adj = sub_adj
        x = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        label = sub_label

        pred_label = torch.argmax(self.pred[neighbors], dim=1)
        print("Node predicted label: ", pred_label[node_idx_new])

        explainer = ExplainModule(
            adj=adj,
            x=x,
            model=self.model,
            label=label,
            graph_mode=self.graph_mode,
        )

        self.model.eval()

        # gradient baseline
        if model == "exp":
            explainer.train()
            begin_time = time.time()
            for epoch in range(self.num_epochs):
                explainer.zero_grad()
                explainer.optimizer.zero_grad()
                ypred = explainer(node_idx_new)
                loss = explainer.loss(ypred, pred_label, node_idx_new, epoch)
                loss.backward()

                explainer.optimizer.step()
                if explainer.scheduler is not None:
                    explainer.scheduler.step()

                mask_density = explainer.mask_density()
                if self.print_training:
                    print(
                        "epoch: ",
                        epoch,
                        "; loss: ",
                        loss.item(),
                        "; mask density: ",
                        mask_density.item(),
                        "; pred: ",
                        ypred,
                    )

            print("finished training in ", time.time() - begin_time)
            if model == "exp":
                masked_adj = (
                        explainer.masked_adj[0].cpu().detach().numpy() * sub_adj.squeeze()
                )

        with open(os.path.join('log', 'Masked Adj.npy'), 'wb') as outfile:
            np.save(outfile, np.asarray(masked_adj.copy()))
            print("Saved adjacency matrix to ", 'Masked Adj.npy')
        return masked_adj

    def explain_graph(
            self, model="exp"
    ):
        """Explain a single node prediction
        """
        # index of the query node in the new adj
        print("val acc before starting: ", accuracy(self.pred[self.test_idx],self.label[self.test_idx]).item())

        x = torch.tensor(self.feat, requires_grad=True, dtype=torch.float)
        best_acc = 0
        best_mask = None

        explainer = ExplainModule(
            adj=self.adj,
            x=x,
            model=self.model,
            label=self.label,
            graph_mode=True,
        )

        pred_label = torch.argmax(self.pred, dim=1)
        self.model.eval()

        # gradient baseline
        if model == "exp":
            explainer.train()
            begin_time = time.time()
            for epoch in range(self.num_epochs):
                explainer.zero_grad()
                explainer.optimizer.zero_grad()
                ypred = explainer(marginalize=False)
                loss = explainer.loss(pred_label[self.train_idx],ypred[self.train_idx,:],self.label[self.train_idx])
                loss.backward()

                explainer.optimizer.step()
                if explainer.scheduler is not None:
                    explainer.scheduler.step()

                mask_density = explainer.mask_density()
                val_acc=accuracy(ypred[self.test_idx], self.label[self.test_idx]).item()

                if self.print_training:
                    print(
                        "epoch: ",
                        epoch,
                        "; loss: ",
                        loss.item(),
                        "; mask density: ",
                        mask_density.item(),
                        "; train acc: ",
                        accuracy(ypred[self.train_idx], self.label[self.train_idx]).item(),
                        "; val acc: ",
                        accuracy(ypred[self.test_idx], self.label[self.test_idx]).item(),
                    )
                if val_acc>best_acc and epoch>5:
                    best_acc=val_acc
                    best_mask=(torch.sigmoid(explainer.feat_mask) if explainer.use_sigmoid else explainer.feat_mask)


            print("finished training in ", time.time() - begin_time)
            if model == "exp":
                masked_adj = (
                        explainer.masked_adj.cpu().detach().numpy() * self.adj.squeeze().detach().numpy()
                )
        #
        # with open(os.path.join('log', 'Masked Adj.npy'), 'wb') as outfile:
        #     np.save(outfile, np.asarray(masked_adj.copy()))
        #     print("Saved adjacency matrix to ", 'Masked Adj.npy')
        # return masked_adj
        feat_mask = (
            torch.sigmoid(explainer.feat_mask) if explainer.use_sigmoid else explainer.feat_mask
        )
        return best_mask,best_acc

    def extract_neighborhood(self, node_idx, graph_idx=0):
        """Returns the neighborhood of a given ndoe."""
        neighbors_adj_row = self.neighborhoods[node_idx, :]
        # index of the query node in the new adj
        node_idx_new = sum(neighbors_adj_row[:node_idx])
        neighbors = np.nonzero(neighbors_adj_row)[0]
        sub_adj = self.adj[neighbors][:, neighbors]
        sub_feat = self.feat[neighbors]
        sub_label = self.label[neighbors]
        return node_idx_new, sub_adj, sub_feat, sub_label, neighbors


class ExplainModule(nn.Module):
    def __init__(
            self,
            adj,
            x,
            model,
            label,
            writer=None,
            use_sigmoid=True,
            graph_mode=False,
    ):
        super(ExplainModule, self).__init__()
        self.adj = adj
        self.x = x
        self.model = model
        self.label = label
        self.mask_act = "sigmoid"
        self.use_sigmoid = use_sigmoid
        self.graph_mode = graph_mode

        init_strategy = "normal"
        num_nodes = adj.size()[1]
        self.mask_bias_flag = True
        self.mask, self.mask_bias = self.construct_edge_mask(
            num_nodes, init_strategy=init_strategy
        )

        self.feat_mask = self.construct_feat_mask(x.size(-1))
        params = [self.mask, self.feat_mask]
        if self.mask_bias_flag is not None:
            params.append(self.mask_bias)
        # For masking diagonal entries
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)

        self.scheduler, self.optimizer = util.build_optimizer(params)

        self.coeffs = {
            "size": 1.0,
            "feat_size": 1.0,
            "ent": 1.0,
            "feat_ent": 1.0,
            "grad": 0,
            "lap": 1.0,
        }

    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(0.5, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 1.0)
        return mask

    def construct_edge_mask(self, num_nodes, init_strategy="normal", const_val=1.0):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
                # mask.clamp_(0.0, 1.0)
        elif init_strategy == "const":
            nn.init.constant_(mask, const_val)

        if self.mask_bias_flag:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias

    def _masked_adj(self):
        sym_mask = self.mask
        if self.mask_act == "sigmoid":
            sym_mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            sym_mask = nn.ReLU()(self.mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.adj
        masked_adj = adj * sym_mask
        if self.mask_bias_flag:
            bias = (self.mask_bias + self.mask_bias.t()) / 2
            bias = nn.ReLU6()(bias * 6) / 6
            masked_adj += (bias + bias.t()) / 2
        return masked_adj * self.diag_mask

    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum

    def feat_mask_density(self):
        mask_sum = torch.mean(self.feat_mask).cpu()
        return mask_sum

    def forward(self, node_idx=0, mask_features=True, marginalize=True):
        x = self.x

        # self.masked_adj = self._masked_adj()
        self.masked_adj = self.adj
        if mask_features:
            feat_mask = (
                torch.sigmoid(self.feat_mask)
                if self.use_sigmoid
                else self.feat_mask
            )
            if marginalize:
                std_tensor = torch.ones_like(x, dtype=torch.float) / 2
                mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
                z = torch.normal(mean=mean_tensor, std=std_tensor)
                x = x + z * (1 - feat_mask)
            else:
                x = x * feat_mask

        ypred = self.model(x, self.masked_adj)
        if self.graph_mode:
            return ypred
        else:
            node_pred = ypred[node_idx, :]
            res = nn.Softmax(dim=0)(node_pred)
            return res

    def adj_feat_grad(self, node_idx, pred_label_node):
        self.model.zero_grad()
        self.adj.requires_grad = True
        self.x.requires_grad = True
        if self.adj.grad is not None:
            self.adj.grad.zero_()
            self.x.grad.zero_()
        x, adj = self.x, self.adj
        ypred, _ = self.model(x, adj)
        if self.graph_mode:
            logit = nn.Softmax(dim=0)(ypred[0])
        else:
            logit = nn.Softmax(dim=0)(ypred[self.graph_idx, node_idx, :])
        logit = logit[pred_label_node]
        loss = -torch.log(logit)
        loss.backward()
        return self.adj.grad, self.x.grad

    def loss(self, pred, pred_label,gt_label=None, node_idx=0):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        if self.graph_mode:
            # pred_label_node = pred_label
            # gt_label_node = self.label
            # logit = pred[gt_label_node]
            # pred_loss = torch.sum(logit)
            pred_loss=F.nll_loss(pred_label, gt_label)
        else:
            mi_obj = False
            if mi_obj:
                pred_loss = -torch.sum(pred * torch.log(pred))
            else:
                pred_label_node = pred_label[node_idx]
                gt_label_node = self.label[node_idx]
                logit = pred[gt_label_node]
                pred_loss = -torch.log(logit)
        # size
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.ReLU()(self.mask)
        size_loss = self.coeffs["size"] * torch.mean(mask)

        # pre_mask_sum = torch.sum(self.feat_mask)
        feat_mask = (
            torch.sigmoid(self.feat_mask) if self.use_sigmoid else self.feat_mask
        )
        feat_size_loss = self.coeffs["feat_size"] * torch.mean(feat_mask)

        # entropy
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)

        feat_mask_ent = - feat_mask \
                        * torch.log(feat_mask) \
                        - (1 - feat_mask) \
                        * torch.log(1 - feat_mask)

        feat_mask_ent_loss = self.coeffs["feat_ent"] * torch.mean(feat_mask_ent)

        # laplacian
        D = torch.diag(torch.sum(self.masked_adj, 0))
        m_adj = self.masked_adj if self.graph_mode else self.masked_adj
        L = D - m_adj
        pred_label_t = torch.tensor(pred_label, dtype=torch.float)
        if self.graph_mode:
            lap_loss = 0
        else:
            lap_loss = (self.coeffs["lap"]
                        * (pred_label_t @ L @ pred_label_t)
                        / self.adj.numel()
                        )

        # grad
        # adj
        # adj_grad, x_grad = self.adj_feat_grad(node_idx, pred_label_node)
        # adj_grad = adj_grad[self.graph_idx]
        # x_grad = x_grad[self.graph_idx]
        # if self.args.gpu:
        #    adj_grad = adj_grad.cuda()
        # grad_loss = self.coeffs['grad'] * -torch.mean(torch.abs(adj_grad) * mask)

        # feat
        # x_grad_sum = torch.sum(x_grad, 1)
        # grad_feat_loss = self.coeffs['featgrad'] * -torch.mean(x_grad_sum * mask)
        # loss = pred_loss  + lap_loss + mask_ent_loss*10 + feat_mask_ent_loss*10+feat_size_loss*5+size_loss*5
        loss = pred_loss  + lap_loss + feat_mask_ent_loss*10+feat_size_loss*5
        # loss = feat_mask_ent_loss * 10 + feat_size_loss * 5

        return loss
