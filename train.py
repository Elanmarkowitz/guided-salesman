"""
Based on https://github.com/tkipf/pygcn
"""
from __future__ import print_function

import time
import argparse
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

from dataprocess import TSPDataset, TSPDirectionDataloader, TSPNeighborhoodDataloader

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--data-dir', type=str, default='data',
                    help='Directory with TSPLIB data.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='Number of training TSP problems per batch')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--train_direction', action='store_true', default=False,
                    help='Indicates to skip training of direction model.')
parser.add_argument('--train_neighborhood', action='store_true', default=False,
                    help='Indicates to skip training of neighborhood model.')


if not args.train_neighborhood and not args.train_direction:
    print("Must provide at least one of '--train_neighborhood' or '--train_direction'")
    exit()

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
dataset = TSPDataset(args.datadir)

if args.train_direction:
    # Model and optimizer
    dir_model = GCN(nfeat=4,  # current node, final node, x-coord, y-coord
                    nhid=args.hidden,
                    nclass=1,  # binary variable
                    dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()


def train(model, dataloader, epochs, optimizer, trainingtype, use_diversity_loss):
    pbar = tqdm(total=epochs*len(dataloader))
    pbar.update(0)
    for e in range(epochs):
        t = time.time()
        model.train()

        for features, full_adj, labels, graph_sizes in dataloader:
            
            optimizer.zero_grad()
            output = model(features, full_adj)

            total_loss = torch.tensor(0.0)
            for start, stop in graph_sizes:
                if trainingtype == 'direction':
                    loss = F.cross_entropy(output[start:stop].T, labels[start:stop].T)
                elif trainingtype == 'neighborhood':
                    loss = F.binary_cross_entropy_with_logits(output[start:stop].T, labels[start:stop].T)
                if use_diversity_loss:
                    loss = torch.min(loss)
                total_loss += loss 

            total_loss.backward()
            optimizer.step()

            if not args.fastmode:
                # Evaluate validation set performance separately,
                # deactivates dropout during validation run.
                model.eval()
                output = model(features, adj)

            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            print('Epoch: {:04d}'.format(e+1),
                'loss_train: {:.4f}'.format(loss_train.item()),
                'acc_train: {:.4f}'.format(acc_train.item()),
                'loss_val: {:.4f}'.format(loss_val.item()),
                'acc_val: {:.4f}'.format(acc_val.item()),
                'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
