import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from gcn import GCN
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import numpy as np
from tqdm import tqdm
import utils
import math
import scipy.sparse as sp
import gc

class BaseMeta(Module):

    def __init__(self, nfeat, hidden_sizes, nclass, nnodes, dropout, train_iters, attack_features, lambda_, device, with_bias=False, lr=0.01, with_relu=False):
        super(BaseMeta, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.nfeat = nfeat
        self.nclass = nclass
        self.with_bias = with_bias
        self.with_relu = with_relu

        self.gcn = GCN(nfeat=nfeat,
                       nhid=hidden_sizes[0],
                       nclass=nclass,
                       dropout=0.5,
                       with_relu=False)


        self.train_iters = train_iters
        self.surrogate_optimizer = optim.Adam(self.gcn.parameters(), lr=lr, weight_decay=5e-4)

        self.attack_features = attack_features
        self.lambda_ = lambda_
        self.device = device
        self.nnodes = nnodes

        self.adj_changes = Parameter(torch.FloatTensor(nnodes, nnodes))
        self.adj_changes.data.fill_(0)

    def filter_potential_singletons(self, modified_adj):
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
        the entry have degree 1 and there is an edge between the two nodes.

        Returns
        -------
        torch.Tensor shape [N, N], float with ones everywhere except the entries of potential singleton nodes,
        where the returned tensor has value 0.

        """

        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()

        l_and = resh * modified_adj
        logical_and_symmetric = l_and + l_and.t()
        flat_mask = 1 - logical_and_symmetric
        return flat_mask

    def train_surrogate(self, features, adj, labels, idx_train, train_iters=200):
        print('=== training surrogate model to predict unlabled data for self-training')
        surrogate = self.gcn
        surrogate.initialize()

        adj_norm = utils.normalize_adj_tensor(adj)
        surrogate.train()
        for i in range(train_iters):
            self.surrogate_optimizer.zero_grad()
            output = surrogate(features, adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            self.surrogate_optimizer.step()

        # Predict the labels of the unlabeled nodes to use them for self-training.
        surrogate.eval()
        output = surrogate(features, adj_norm)
        labels_self_training = output.argmax(1)
        labels_self_training[idx_train] = labels[idx_train]
        # reset parameters for later updating
        surrogate.initialize()
        return labels_self_training


    def log_likelihood_constraint(self, modified_adj, ori_adj, ll_cutoff):
        """
        Computes a mask for entries that, if the edge corresponding to the entry is added/removed, would lead to the
        log likelihood constraint to be violated.
        """

        t_d_min = torch.tensor(2.0).to(self.device)
        t_possible_edges = np.array(np.triu(np.ones((self.nnodes, self.nnodes)), k=1).nonzero()).T
        allowed_mask, current_ratio = utils.likelihood_ratio_filter(t_possible_edges,
                                                                    modified_adj,
                                                                    ori_adj, t_d_min,
                                                                    ll_cutoff)

        return allowed_mask, current_ratio


class AtkSE(BaseMeta):

    def __init__(self, args, nfeat, hidden_sizes, nclass, nnodes, dropout, train_iters,
                 attack_features, device, lambda_=0.5, with_relu=False, with_bias=False, lr=0.1, momentum=0.9):

        super(AtkSE, self).__init__(nfeat, hidden_sizes, nclass, nnodes, dropout, train_iters, attack_features, lambda_, device, with_bias=with_bias, with_relu=with_relu)

        self.momentum = momentum
        self.lr = lr

        self.weights = []
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []
        self.momentum_grad = None
        self.mom = 0.7

        previous_size = nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(device))
            bias = Parameter(torch.FloatTensor(nhid).to(device))
            w_velocity = torch.zeros(weight.shape).to(device)
            b_velocity = torch.zeros(bias.shape).to(device)
            previous_size = nhid

            self.weights.append(weight)
            self.biases.append(bias)
            self.w_velocities.append(w_velocity)
            self.b_velocities.append(b_velocity)

        output_weight = Parameter(torch.FloatTensor(previous_size, nclass).to(device))
        output_bias = Parameter(torch.FloatTensor(nclass).to(device))
        output_w_velocity = torch.zeros(output_weight.shape).to(device)
        output_b_velocity = torch.zeros(output_bias.shape).to(device)

        self.weights.append(output_weight)
        self.biases.append(output_bias)
        self.w_velocities.append(output_w_velocity)
        self.b_velocities.append(output_b_velocity)

        self.dropnode = args.dropnode
        self.gauss_noise = args.gauss_noise
        self.smooth_loop = args.smooth_loop
        self.wait_list = args.wait_list
        self.intervals = args.intervals
        self.candidates = args.candidates

        self._initialize()

    def _initialize(self):

        for w, b in zip(self.weights, self.biases):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)


    def inner_train(self, features, adj_norm, idx_train, idx_unlabeled, labels):
        self._initialize()

        for ix in range(len(self.hidden_sizes) + 1):
            self.weights[ix] = self.weights[ix].detach()
            self.weights[ix].requires_grad = True
            self.w_velocities[ix] = self.w_velocities[ix].detach()
            self.w_velocities[ix].requires_grad = True

            if self.with_bias:
                self.biases[ix] = self.biases[ix].detach()
                self.biases[ix].requires_grad = True
                self.b_velocities[ix] = self.b_velocities[ix].detach()
                self.b_velocities[ix].requires_grad = True

        for j in range(self.train_iters):
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b
                if self.with_relu:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])

            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]

    def get_meta_grad(self, features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training,iters):

        hidden = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu:
                hidden = F.relu(hidden)
        
        output = F.log_softmax(hidden/1000, dim=1)
        attack_loss = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])
        adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]

        return adj_grad


    def forward(self, features, ori_adj, labels, idx_train, idx_unlabeled, perturbations):
        self.sparse_features = sp.issparse(features)

        adj_norm = utils.normalize_adj_tensor(ori_adj)
        self.inner_train(features, adj_norm, idx_train, idx_unlabeled, labels)
        hidden = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu:
                hidden = F.relu(hidden)
        
        output = F.log_softmax(hidden, dim=1)
        labels_self_training = output.argmax(1)
        labels_self_training[idx_train] = labels[idx_train]

        iters = 0
        for i in tqdm(range(perturbations), desc="Perturbing graph"):
            iters = iters+1
            adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
            ind = np.diag_indices(self.adj_changes.shape[0])
            adj_changes_symm = torch.clamp(adj_changes_square + torch.transpose(adj_changes_square, 1, 0), -1, 1)
            modified_adj = adj_changes_symm + ori_adj

            adj_norm = utils.normalize_adj_tensor(modified_adj)
            self.inner_train(features, adj_norm, idx_train, idx_unlabeled, labels)
            adj_grad = self.get_meta_grad(features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training,iters)

            smoothed_grad = 0
            smoothed_grad = smoothed_grad + adj_grad
            
            for loop in range(self.smooth_loop):
                noise = torch.normal(0, self.gauss_noise, size=(features.shape[0], features.shape[1])).to(self.device)
                adj_grad = self.get_meta_grad(features+noise, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training, True)
                smoothed_grad = smoothed_grad + 0.2*adj_grad/self.smooth_loop

            adj_meta_grad = smoothed_grad * (-2 * modified_adj + 1)
            adj_meta_grad -= adj_meta_grad.min()
            adj_meta_grad -= torch.diag(torch.diag(adj_meta_grad, 0))
            singleton_mask = self.filter_potential_singletons(modified_adj)
            adj_meta_grad = adj_meta_grad *  singleton_mask

            if self.momentum_grad is not None:
                self.momentum_grad = adj_meta_grad.detach() + self.mom*self.momentum_grad
            else:
                self.momentum_grad = adj_meta_grad.detach()

            structural_grad = self.momentum_grad + self.momentum_grad.t()
            structural_grad = structural_grad.reshape([structural_grad.shape[0]*structural_grad.shape[1]])
            grad_abs = torch.abs(structural_grad)

            idxes = torch.argsort(grad_abs, dim=0, descending=True)
            candidate_list = []
            wait_list = []
            grad_sum = []
            _N = smoothed_grad.shape[0]
            for index in range(idxes.shape[0]):
                x = idxes[index] // _N
                x=x.item()
                y = idxes[index] % _N
                y=y.item()
                if (x*_N+y) in candidate_list or (y*_N+x) in candidate_list:
                    continue
                wait_list.append([x,y])
                candidate_list.append(idxes[index].item())

                candidate_adj = modified_adj.clone()
                if len(wait_list) == self.wait_list:
                    interval = self.intervals
                    steps = int(1/interval)
                    grad_list = [interval*k for k in range(steps)]
                    
                    cand_grad = []
                    for grad_weight in grad_list:
                        for candidate in wait_list:
                            if modified_adj[candidate[0],candidate[1]] == 0:
                                candidate_adj[candidate[0],candidate[1]] = grad_weight
                                candidate_adj[candidate[1],candidate[0]] = grad_weight
                            else:
                                candidate_adj[candidate[0],candidate[1]] = 1 - grad_weight
                                candidate_adj[candidate[1],candidate[0]] = 1 - grad_weight

                        adj_norm = utils.normalize_adj_tensor(candidate_adj)
                        adj_grad = self.get_meta_grad(features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training, True)
                        adj_grad = adj_grad * (-2 * modified_adj + 1)

                        cand_grad1 = []
                        cand_grad2 = []
                        for candidate in wait_list:
                            cand_grad1.append(adj_grad[candidate[0],candidate[1]].item())
                            cand_grad2.append(adj_grad[candidate[1],candidate[0]].item())
                        del adj_grad
                        gc.collect()
                        cand_grad.append([cand_grad1,cand_grad2])

                    for candidate_idx,candidate in enumerate(wait_list):
                        cand_grad_sum = 0
                        if modified_adj[candidate[0],candidate[1]] == 0:
                            for step in range(steps):
                                cand_grad_sum = cand_grad_sum + interval*(cand_grad[step][0][candidate_idx]) + interval*(cand_grad[step][1][candidate_idx])
                        else:
                            for step in range(steps):
                                cand_grad_sum = cand_grad_sum - interval*(cand_grad[step][0][candidate_idx]) - interval*(cand_grad[step][1][candidate_idx])
                        grad_sum.append(cand_grad_sum)

                    wait_list = []

                    if len(candidate_list)>=self.candidates:
                        order = np.argsort(grad_sum)
                        order = order[::-1]
                        candidate_list = [candidate_list[new_ord] for new_ord in order]       
                        break

            
            row_idx = candidate_list[0] // _N
            col_idx = candidate_list[0] % _N

            self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)

            if self.attack_features:
                pass

        return self.adj_changes + ori_adj

