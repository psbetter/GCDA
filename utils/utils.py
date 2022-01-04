import random

import torch
from torch.autograd import Function
import numpy as np
from easydict import EasyDict as edict
import yaml

def Config(filename):
    with open(filename, 'r') as f:
        parser = edict(yaml.load(f, Loader=yaml.FullLoader))
    for x in parser:
        print('{}: {}'.format(x, parser[x]))
    return parser

def lr_scheduler(optimizer, init_lr, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def print_args(args):
    message = ['src_address', 'tgt_address', 'src_ns_address']
    log_str = ("================= start ==================\n")
    for arg, content in args.__dict__.items():
        if arg not in message:
            continue
        if args.setting == 'UDA' and arg == 'src_ns_address':
            continue
        log_str += ("{}:{}\n".format(arg, content))
    print(log_str)
    args.out_file.write(log_str+'\n')
    args.out_file.flush()

class infoNCE():
    def __init__(self, features=None, labels=None, class_num=10, feature_dim=512):
        super(infoNCE, self).__init__()
        self.features = features
        self.labels = labels
        self.class_num = class_num

    def get_posAndneg(self, features, labels, tgt_label=None, feature_q_idx=None, co_fea=None):
        self.features = features
        self.labels = labels

        # get the label of q
        q_label = tgt_label[feature_q_idx]

        # get the positive sample
        positive_sample_idx = []
        for i, label in enumerate(self.labels):
            if label == q_label:
                positive_sample_idx.append(i)

        if len(positive_sample_idx) != 0:
            feature_pos = self.features[random.choice(positive_sample_idx)].unsqueeze(0)
        else:
            feature_pos = co_fea.unsqueeze(0)


        # get the negative samples
        negative_sample_idx = []
        for idx in range(features.shape[0]):
            if self.labels[idx] != q_label:
                negative_sample_idx.append(idx)

        negative_pairs = torch.Tensor([]).cuda()
        for i in range(self.class_num - 1):
            negative_pairs = torch.cat((negative_pairs, self.features[random.choice(negative_sample_idx)].unsqueeze(0)))
        if negative_pairs.shape[0] == self.class_num - 1:
            features_neg = negative_pairs
        else:
            raise Exception('Negative samples error!')

        return torch.cat((feature_pos, features_neg))