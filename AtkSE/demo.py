import torch
from gcn import GCN
from utils import *
import argparse
import numpy as np
from attack import AtkSE
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=18, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dataset', type=str, default='cora', help='dataset')
parser.add_argument('--momentum', type=float, default=0.9, help='model variant')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')

parser.add_argument('--dropnode', type=float, default=0.05)
parser.add_argument('--gauss_noise', type=float, default=2e-4)
parser.add_argument('--smooth_loop', type=int, default=40)
parser.add_argument('--wait_list', type=int, default=4)
parser.add_argument('--intervals', type=float, default=0.25)
parser.add_argument('--candidates', type=int, default=32)

args = parser.parse_args()

cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

# === loading dataset
adj, features, labels = load_data(dataset=args.dataset)
nclass = max(labels) + 1

val_size = 0.1
test_size = 0.8
train_size = 1 - test_size - val_size

idx = np.arange(adj.shape[0])
idx_train, idx_val, idx_test = get_train_val_test(idx, train_size, val_size, test_size, stratify=labels)
idx_unlabeled = np.union1d(idx_val, idx_test)
perturbations = int(args.ptb_rate * (adj.sum()//2))

adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)


model = AtkSE(args,nfeat=features.shape[1], hidden_sizes=[args.hidden],
                    nnodes=adj.shape[0], nclass=nclass, dropout=0.5,
                    train_iters=100, attack_features=False, lambda_=0, device=device, momentum=args.momentum)

if device != 'cpu':
    adj = adj.to(device)
    features = features.to(device)
    labels = labels.to(device)
    model = model.to(device)

def main():

    modified_adj = model(features, adj, labels, idx_train,
                        idx_unlabeled, perturbations)
    modified_adj = modified_adj.detach()
    np.savetxt('GraD_modified_{}_{}.txt'.format(args.dataset,args.ptb_rate),modified_adj.cpu().numpy())

if __name__ == '__main__':
    main()

