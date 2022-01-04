import torch.nn as nn
from torch.autograd import Variable

import model.backbone as backbone
import torch
import torch.nn.functional as F
import numpy as np
from model import loss
from utils.tools import update_mem_feat, neighbor_prototype, update_center, update_prototype


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class Resnet(nn.Module):
    def __init__(self, base_net='ResNet50'):
        super(Resnet, self).__init__()
        ## set base network
        self.base_network = backbone.network_dict[base_net]()
        self.output_num = self.base_network.output_num()

    def forward(self, inputs):
        features = self.base_network(inputs)

        return features

class Classifier(nn.Module):
    def __init__(self, base_output_num=256, cfg=None):
        super(Classifier, self).__init__()
        self.bottleneck_layer = nn.Linear(base_output_num, cfg.bottleneck_dim)
        self.fc2 = nn.Linear(cfg.bottleneck_dim, cfg.class_num, bias=False)
        self.class_num = cfg.class_num
        self.temp = cfg.temp
        self.setting = cfg.setting
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.setting == 'NIUDA':
            self.fc1 = nn.Linear(cfg.bottleneck_dim, cfg.class_num, bias=False)
            self.share_class_num = cfg.share_class_num
            mask = [1 if i < self.share_class_num else 0 for i in range(self.class_num)]
            self.mask = torch.ByteTensor(mask).cuda()
            self.centroid = torch.zeros(self.class_num, cfg.bottleneck_dim).cuda()
    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        features = self.bottleneck_layer(x)
        # MFT

        # CDA
        outputs2 = self.fc2(features)
        softmax_outputs = F.softmax(outputs2, dim=1)
        return features, outputs2, softmax_outputs

class model(object):
    def __init__(self, cfg, use_gpu=True):
        self.base_net = Resnet(cfg.backbone)
        self.classifier = Classifier(self.base_net.output_num, cfg)

        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = cfg.class_num
        if self.use_gpu:
            self.base_net = self.base_net.cuda()
            self.classifier = self.classifier.cuda()
        self.bottleneck_dim = cfg.bottleneck_dim
        self.class_weight_src = torch.ones(cfg.class_num, ).cuda()
        self.smooth = cfg.smooth
        self.setting = cfg.setting
        self.prototype_feat_s = torch.rand(self.class_num, self.bottleneck_dim).cuda()
    # init MIM
    def init_mom_softmax(self, train_loader_len):
        self.momentum_softmax_target = loss.MomentumSoftmax(
            self.class_num, m=train_loader_len)

    def get_loss(self, inputs, labels_source, idx_s, idx_t):
        bs = labels_source.size(0)

        features_ = self.base_net(inputs)
        features, outputs2, _ = self.classifier(features_)
        feature_source = features.narrow(0, 0, bs)
        outputs_source = outputs2.narrow(0, 0, bs)

        feature_target = features.narrow(0, bs, inputs.size(0) - bs)
        outputs_target = outputs2.narrow(0, bs, inputs.size(0) - bs)

        # update prototype_feat_s
        up_feat_s = update_prototype(feature_source, labels_source, self.prototype_feat_s)
        self.prototype_feat_s = up_feat_s.detach()

        # label smooth regular loss
        src_ = loss.CrossEntropyLabelSmooth(reduction='none', num_classes=self.class_num, epsilon=self.smooth)(
            outputs_source, labels_source)
        weight_src = self.class_weight_src[labels_source].unsqueeze(0)
        classifier_loss = torch.sum(weight_src * src_) / (torch.sum(weight_src).item())
        # classifier_loss += nn.CrossEntropyLoss()(outputs_source, labels_share)

        # contrastive loss
        con_loss = loss.contrastive_loss(feature_target, self.prototype_feat_s, t=2)

        # MIM loss
        min_entroy_loss = loss.entroy_mim(outputs_target)
        prob_unl = F.softmax(outputs_target, dim=1)
        prob_mean_unl = prob_unl.sum(dim=0) / outputs_target.size(0)
        self.momentum_softmax_target.update(prob_mean_unl.cpu().detach(), outputs_target.size(0)) # update momentum
        momentum_prob_target = (self.momentum_softmax_target.softmax_vector.cuda()) # get momentum probability
        entropy_cond = -torch.sum(prob_mean_unl * torch.log(momentum_prob_target + 1e-5))
        max_entroy_loss = -entropy_cond

        self.iter_num += 1


        return classifier_loss, con_loss, min_entroy_loss, max_entroy_loss

    def predict(self, inputs):
        features = self.base_net(inputs)
        _, _, softmax_outputs = self.classifier(features)
        return softmax_outputs

    def set_train(self, mode):
        self.base_net.train(mode)
        self.classifier.train(mode)
        self.is_train = mode

    def get_visual_feature(self, inputs, labels, index):
        feature_ = self.base_net(inputs)
        features, _, _, _, softmax_outputs = self.classifier(feature_)
        if index == 1:
            # with open('./CMFT_label_aw2.tsv', 'ab') as f:
            #     labels1 = labels.detach().cpu().numpy()
            #     np.savetxt(f, labels1, delimiter='\t')
            with open('./CMFT_embed_ac.tsv', 'ab') as f:
                features2 = features.detach().cpu().numpy()
                np.savetxt(f, features2, delimiter='\t')
            with open('./CMFT_label_ac.tsv', 'ab') as f:
                labels = labels.unsqueeze(1)
                labels_t = torch.zeros_like(labels)

                labels_final = torch.cat((labels, labels_t), dim=1)
                labels_final = labels_final.detach().cpu().numpy()
                np.savetxt(f, labels_final, delimiter='\t')
        else:
            with open('./CMFT_embed_ac.tsv', 'ab') as f:
                features = features.detach().cpu().numpy()
                np.savetxt(f, features, delimiter='\t')
            with open('./CMFT_label_ac.tsv', 'ab') as f:
                labels = labels.unsqueeze(1)
                labels_s = torch.ones_like(labels)
                labels_final = torch.cat((labels, labels_s), dim=1)
                labels_final = labels_final.detach().cpu().numpy()
                np.savetxt(f, labels_final, delimiter='\t')
            # with open('./CMFT_label_aw2.tsv', 'ab') as f:
            #     labels1 = labels.detach().cpu().numpy()
            #     np.savetxt(f, labels1, delimiter='\t')
        return softmax_outputs