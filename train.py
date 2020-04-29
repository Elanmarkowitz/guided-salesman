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
from pygcn.multi_model import MultiModel

from dataprocess import TSPDataset, TSPDirectionDataloader, TSPNeighborhoodDataloader

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--datadir', type=str, default='data',
                    help='Directory with TSPLIB data.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Number of training TSP problems per batch')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--train_direction', action='store_true', default=False,
                    help='Indicates to skip training of direction model.')
parser.add_argument('--train_neighborhood', action='store_true', default=False,
                    help='Indicates to skip training of neighborhood model.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not args.train_neighborhood and not args.train_direction:
    print("Must provide at least one of '--train_neighborhood' or '--train_direction'")
    exit()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
dataset = TSPDataset(args.datadir)


def train(model, dataloader, epochs, optimizer, trainingtype, use_diversity_loss=True):
    if args.cuda:
        model.cuda()
    pbar = tqdm(total=epochs*len(dataloader))
    pbar.update(0)
    for e in range(epochs):
        t = time.time()
        model.train()

        for features, full_adj, labels, graph_sizes in dataloader:
            pbar.update(1)
            if args.cuda:
                features = features.cuda()
                full_adj = full_adj.cuda()
                labels = [l.cuda() for l in labels]
            
            optimizer.zero_grad()
            output = model(features, full_adj)

            total_loss = torch.tensor(0.0)
            if args.cuda:
                total_loss = total_loss.cuda()
            for (start, stop), label in zip(graph_sizes, labels):
                if trainingtype == 'direction':
                    out_dir = output[:,start:stop].mean(1)
                    d = (out_dir - label).norm(p=2, dim=1)
                    loss = d**2
                elif trainingtype == 'neighborhood':
                    outpreds = output[:,start:stop].squeeze(-1)
                    labs = labels[start:stop].expand_as(outpreds)
                    loss = F.binary_cross_entropy_with_logits(outpreds, labs)
                if use_diversity_loss:
                    loss = torch.min(loss)
                else:
                    loss = torch.sum(loss)
                total_loss += loss 

            pbar.write(f'Loss: {total_loss.cpu().detach().numpy()}')

            total_loss.backward()
            optimizer.step()

        # if not args.fastmode:
        #     # Evaluate validation set performance separately,
        #     # deactivates dropout during validation run.
        #     model.eval()
        #     output = model(features, adj)

        # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        # acc_val = accuracy(output[idx_val], labels[idx_val])
        # pbar.write('Epoch: {:04d}'.format(e+1),
        #            'loss_train: {:.4f}'.format(total_loss.item()),
        #            'acc_train: {:.4f}'.format(acc_train.item()),
        #            'loss_val: {:.4f}'.format(loss_val.item()),
        #            'acc_val: {:.4f}'.format(acc_val.item()),
        #            'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


if args.train_direction:
    # Model and optimizer
    dir_model = MultiModel(GCN, 3, 
                           nfeat=6,  # current node, final node, final coords (x,y), x-coord, y-coord
                           nhid=args.hidden,
                           nclass=2,  # direction 
                           dropout=args.dropout)
    optimizer = optim.Adam(dir_model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    dir_dataloader = TSPDirectionDataloader(dataset, batch_size=args.batch_size, shuffle=True)
    train(dir_model, dir_dataloader, args.epochs, optimizer, 'direction', use_diversity_loss=True)


if args.train_neighborhood:
    nbr_model = MultiModel(GCN, 1, 
                           nfeat=8,  # current node, final node, final-node coords, x-coord, y-coord, dir-node coords
                           nhid=args.hidden,
                           nclass=1,  # binary, in neighborhood 
                           dropout=args.dropout)
    optimizer = optim.Adam(nbr_model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    nbr_dataloader = TSPNeighborhoodDataloader(dataset, batch_size=args.batch_size, shuffle=True)
    train(nbr_model, nbr_dataloader, args.epochs, optimizer, 'neighborhood', use_diversity_loss=True)

import pdb; pdb.set_trace()

# Train model
# t_total = time.time()
# for epoch in range(args.epochs):
#     train(epoch)
# print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
# test()
