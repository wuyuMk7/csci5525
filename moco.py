#!/usr/bin/env python

from functools import partial
from torchvision.models import resnet
import torch
import torch.nn as nn
import copy
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'vit-pytorch')))
from vit_pytorch import ViT

class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.net(x)
        return x

class ModelBase(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """
    def __init__(self, feature_dim=128, arch=None, bn_splits=16, ver=1):
        super(ModelBase, self).__init__()

        # use split batchnorm
        norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        self.net = []
        for name, module in net.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))
            self.net.append(module)

        if ver != 1:
            # self.net.append(self.net[-1])
            # self.net[-2] = MLP(in_dim=self.net[-1].in_features, out_dim=self.net[-1].in_features)
            fc_in_dim = self.net[-1].in_features
            self.net[-1] = nn.Sequential(nn.Linear(fc_in_dim, fc_in_dim), nn.ReLU(), self.net[-1])

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.net(x)
        # note: not normalized here
        return x

class ViTBase(nn.Module):
    def __init__(self, moco_dim):
        super(ViTBase, self).__init__()

        # use split batchnorm
        # norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
        # resnet_arch = getattr(resnet, arch)
        # net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        vit_dim = 256
        proj_hid_dim = 1024

        vit = ViT(
            image_size = 32,
            patch_size = 4,
            num_classes = moco_dim,
            # dim = 256,
            # depth = 4,
            # heads = 12,
            # mlp_dim = 512,
            dim = vit_dim,
            depth = 3,
            heads = 8,
            mlp_dim = 384,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        # net = []
        # for name, module in vit.named_children():
        #     if name != "mlp_head":
        #         net.append(module)
        # self.net = nn.Sequential(*net)
        self.net = vit

        self.net.mlp_head = nn.Sequential(
            nn.Linear(vit_dim, proj_hid_dim),
            nn.BatchNorm1d(proj_hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hid_dim, proj_hid_dim),
            nn.BatchNorm1d(proj_hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hid_dim, moco_dim)
        )
        self.net.mlp_head.requires_grad = False

    def forward(self, x):
        x = self.net(x)
        # print(x.shape)
        # exit(0)
        return x


class MoCo(nn.Module):
    def __init__(self, dim=128, K=4096, m=0.99, T=0.1, ver=1, arch='resnet18', bn_splits=8, symmetric=True, v3_encoder=None):
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric

        # create the encoders
        if ver == 3:
            self.ver3 = True

            # self.encoder_q = ViTBase(copy.deepcopy(v3_encoder))
            # self.encoder_k = ViTBase(copy.deepcopy(v3_encoder))
            # self.encoder_q = copy.deepcopy(v3_encoder)
            # self.encoder_k = copy.deepcopy(v3_encoder)
            self.encoder_q = ViTBase(dim)
            self.encoder_k = ViTBase(dim)
            self.predictor = nn.Sequential(
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
            )
        else:
            self.ver3 = False
            self.encoder_q = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits, ver=ver)
            self.encoder_k = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits, ver=ver)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, im_q, im_k):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss, q, k

    def v3_loss(self, imq, imk):
        # compute query features
        q = self.predictor(self.encoder_q(imq))  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(imk)

            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        logits = torch.mm(q, k.t())
        N = q.size(0)
        labels = range(N)
        labels = torch.LongTensor(labels).cuda()
        loss = nn.CrossEntropyLoss().cuda()(logits / self.T, labels)
        return 2 * self.T * loss

    def forward(self, im1, im2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        if self.ver3:
            loss_qk = self.v3_loss(im1, im2)
            loss_kq = self.v3_loss(im2, im1)
            loss = loss_qk + loss_kq
        else:
            # compute loss
            if self.symmetric:  # symmetric loss, for v3
                loss_12, q1, k2 = self.contrastive_loss(im1, im2)
                loss_21, q2, k1 = self.contrastive_loss(im2, im1)
                loss = loss_12 + loss_21
                k = torch.cat([k1, k2], dim=0)
            else:  # asymmetric loss
                loss, q, k = self.contrastive_loss(im1, im2)

            self._dequeue_and_enqueue(k)

        return loss
