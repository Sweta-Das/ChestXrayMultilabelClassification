# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.utils.weight_init import trunc_normal_

from ..builder import HEADS
from .cls_head import ClsHead

import torch


@HEADS.register_module()
class My_Hybird_Head(ClsHead):
    def __init__(
            self,
            num_classes,
            in_channels,  # [conv_dim, trans_dim]
            init_cfg=dict(type='Normal', layer='Linear', std=0.01),
            *args,
            **kwargs):
        super(My_Hybird_Head, self).__init__(init_cfg=None, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.init_cfg = init_cfg

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')
        # in_channels [256, 384]
        self.conv_cls_head = nn.Linear(self.in_channels[0],
                                       num_classes)
        self.trans_cls_head = nn.Linear(self.in_channels[1],
                                        num_classes) 
        # Learnable fusion layer to combine the two branch logits
        self.fc = nn.Linear(
            in_features=num_classes * 2, out_features=num_classes)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights(self):
        super(My_Hybird_Head, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return
        else:
            self.apply(self._init_weights)

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def simple_test(self, x, sigmoid=True, post_process=True):
        x = self.pre_logits(x)
        assert len(x) == 2
        conv_cls_score = self.conv_cls_head(x[0])
        tran_cls_score = self.trans_cls_head(x[1]) 
        cls_score = self.fc(torch.cat([conv_cls_score, tran_cls_score], dim=1))
        if sigmoid:
            pred = (
                F.sigmoid(cls_score) if cls_score is not None else None)  # pred: tensor(32,7)
            if post_process:  # post_process： True
                pred = self.post_process(pred)
        else:
            pred = cls_score
            if post_process:
                pred = self.post_process(pred)
        return pred

    def forward_train(self, x, gt_label, **kwargs):  # gt_label [32, 11]
        x = self.pre_logits(x)
        assert isinstance(x, list) and len(x) == 2, \
            'There should be two outputs in the CTransCNN model'
        conv_cls_score = self.conv_cls_head(x[0])
        tran_cls_score = self.trans_cls_head(x[1])
        cls_score = self.fc(torch.cat([conv_cls_score, tran_cls_score], dim=1))

        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses

    def loss(self, cls_score, gt_label, **kwargs):
        gt_label = gt_label.type_as(cls_score)
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss = self.compute_loss(
            cls_score, gt_label, avg_factor=num_samples, **kwargs)
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses['accuracy'] = {
                f'top-{k}': a
                for k, a in zip(self.topk, acc)
            }
        losses['loss'] = loss
        return losses
