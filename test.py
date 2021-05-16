#!/usr/bin/env python

from datetime import datetime
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
import torch

import argparse
import sys
import os
import json
import math

from data import load_train, load_test, load_memory
from moco import MoCo

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'vit-pytorch')))
from vit_pytorch import ViT

# test using a knn monitor
def test(net, memory_data_loader, test_data_loader, epoch, args):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

    return total_top1 / total_num * 100

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MoCo on CIFAR-10')

    parser.add_argument('-a', '--arch', default='resnet18')
    parser.add_argument('-v', '--version', default=1, type=int, help='MoCo version')
    parser.add_argument('--workers', default=16, type=int, help='number of workers')

    # lr: 0.06 for batch 512 (or 0.03 for batch 256)
    parser.add_argument('--lr', '--learning-rate', default=0.06, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--epochs', default=1, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

    parser.add_argument('--batch-size', default=512, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')

    # moco specific configs:
    parser.add_argument('--moco-dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--moco-k', default=4096, type=int, help='queue size; number of negative keys')
    parser.add_argument('--moco-m', default=0.99, type=float, help='moco momentum of updating key encoder')
    parser.add_argument('--moco-t', default=0.1, type=float, help='softmax temperature')

    parser.add_argument('--bn-splits', default=8, type=int, help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')

    parser.add_argument('--symmetric', action='store_true', help='use a symmetric loss function that backprops to both crops')

    # knn monitor
    parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
    parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')

    # utils
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')

    '''
    args = parser.parse_args()  # running in command line
    '''
    # args = parser.parse_args('')  # running in ipynb
    args = parser.parse_args()  # running in command line

    # set command line arguments here when running in ipynb
    # args.epochs = 200
    args.cos = False
    args.schedule = []  # cos in use
    args.symmetric = False
    if args.results_dir == '':
        args.results_dir = './results/cache-ver{}-{}'.format(args.version, datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco"))

    if args.version == 2:
        args.cos = True
        args.moco_t=0.2
    if args.version == 3:
        args.cos = True
        args.symmetric = True

    vit = ViT(
        image_size = 32,
        patch_size = 4,
        num_classes = args.moco_dim,
        dim = 256,
        depth = 4,
        heads = 16,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    model = MoCo(
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        ver=args.version,
        arch=args.arch,
        bn_splits=args.bn_splits,
        symmetric=args.symmetric,
        v3_encoder=vit
    ).cuda()

    train_data, train_loader = load_train(args)
    memory_data, memory_loader = load_memory(args)
    test_data, test_loader = load_test(args)
    # load model if resume
    epoch_start = 1

    if args.resume != '':
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        epoch_start = checkpoint['epoch'] + 1
        print('Loaded from: {}'.format(args.resume))

    test_acc_1 = test(model.encoder_q, memory_loader, test_loader, 1, args)
    print(test_acc_1)

