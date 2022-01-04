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
    def __init__(self, base_output_num=256, cfg=None, symmetric_graph=None, node_num=87):
        super(Classifier, self).__init__()
        self.bottleneck_layer = nn.Linear(base_output_num, cfg.bottleneck_dim)
        self.fc = nn.Linear(cfg.bottleneck_dim, cfg.class_num, bias=False)
        # self.bottleneck_layer2 = nn.Linear(base_output_num, cfg.bottleneck_dim)
        self.fc2 = nn.Linear(cfg.bottleneck_dim, cfg.class_num, bias=False)
        self.class_num = cfg.class_num
        self.temp = cfg.temp
        self.setting = cfg.setting
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.centroid = torch.zeros(cfg.class_num, cfg.bottleneck_dim).cuda()

        # set HGR
        self.d_o = 2048
        self.d_l = 128
        self.d_c = 128
        self.m = node_num
        self.k = 128

        self.sgr_conv_o = nn.Sequential(nn.Conv2d(self.d_o, 4 * self.d_l, kernel_size=1),
                                        nn.Conv2d(4 * self.d_l, self.d_l, kernel_size=1))
        self.sgr_conv_a = nn.Conv2d(self.d_l, self.m, kernel_size=1)
        self.sgr_softmax_a = nn.Softmax(dim=-2)
        self.sgr_conv_ps = nn.Conv2d(self.d_l, self.d_c, kernel_size=1)
        self.sgr_relu_m1 = nn.ReLU(inplace=True)
        self.sgr_bn_m1 = nn.BatchNorm1d(self.d_c)
        self.sgr_node = nn.Parameter(torch.Tensor(self.k, self.m), requires_grad=True)
        self.sgr_node.data.uniform_(-0.1, 0.1)
        self.edge = torch.from_numpy(symmetric_graph).float().cuda()
        self.edge_norm = self.norm_adj_matrix(symmetric_graph)
        self.sgr_conv_g = nn.Conv1d(self.d_c + self.k, self.d_c, kernel_size=1)
        self.sgr_relu_m2 = nn.ReLU(inplace=True)
        self.sgr_bn_m2 = nn.BatchNorm1d(self.d_c)
        self.sgr_conv_sp = nn.Conv1d(self.d_c, self.d_l, kernel_size=1)
        self.sgr_conv_s = nn.Conv2d(self.d_c + self.d_l, 1, kernel_size=1)
        self.sgr_softmax_s = nn.Softmax(dim=-2)
        self.sgr_relu_m3 = nn.ReLU(inplace=True)
        self.sgr_bn_m3 = nn.BatchNorm1d(self.d_c)

        self.sgr_conv_as = nn.Conv2d(self.d_l, self.k, kernel_size=1)
        self.sgr_softmax_as = nn.Softmax(dim=2)
        self.sgr_conv_nodes = nn.Conv1d(2 * self.k, self.k, kernel_size=1)
        self.sgr_conv_ox = nn.Sequential(nn.Conv2d(self.d_l, 4 * self.d_l, kernel_size=1),
                                         nn.Conv2d(4 * self.d_l, self.d_o, kernel_size=1))

    def forward(self, x):
        batch_size = x.size(0)
        hl = x.size(2)
        wl = x.size(3)

        x_p = self.avgpool(x)
        x_p = x_p.view(x_p.size(0), -1)
        features_p = self.bottleneck_layer(x_p)
        # HGR
        pre_output = self.fc(features_p)  # [batch,300]
        pre_output2 = pre_output.clone().detach()
        pre_score = F.softmax(pre_output2, dim=1)

        # todo :another way sum = 1
        tree_node = torch.zeros(batch_size, self.m - self.class_num).cuda()
        v_y_init = torch.cat((tree_node, pre_score), dim=1)
        v_y_init = v_y_init.view(batch_size, self.m, -1)

        v_y = self.pagerank_power_batch(self.edge, v_y_init)
        x_l = self.sgr_conv_o(x)
        a = self.sgr_conv_a(x_l)
        a = a.view(batch_size, self.m, -1).permute(0, 2, 1)
        ps = self.sgr_conv_ps(x_l)
        ps = ps.view(batch_size, self.d_c, -1)
        m1 = torch.matmul(ps, a)
        m1 = self.sgr_relu_m1(m1)

        node = self.sgr_node.unsqueeze(0).expand(batch_size, self.k, self.m)

        node = node.contiguous().view(self.k, -1)
        v_y = v_y.view(-1)

        node = v_y * node
        node = node.view(batch_size, self.k, -1)

        g = self.sgr_conv_g(torch.cat((node, m1), 1))
        edge = self.edge_norm.unsqueeze(0).expand(batch_size, self.m, self.m)
        m2 = torch.matmul(g, edge)
        m2 = self.sgr_relu_m2(m2)
        sp = self.sgr_conv_sp(m2)
        m2_expand = m2.unsqueeze(2).expand(batch_size, self.d_c, hl * wl, self.m)
        x_expand = x_l.view(batch_size, self.d_l, -1).unsqueeze(-1).expand(batch_size, self.d_l, hl * wl, self.m)
        s = self.sgr_conv_s(torch.cat((m2_expand, x_expand), 1))
        s = s.squeeze(dim=1).permute(0, 2, 1)
        s = self.sgr_softmax_s(s)
        m3 = torch.matmul(sp, s)
        m3 = self.sgr_relu_m3(m3)
        m3 = m3.view(batch_size, self.d_l, hl, wl)
        m3 = self.sgr_conv_ox(m3)
        # x = x + m3

        feature_aug = self.avgpool(m3)
        feature_aug = feature_aug.view(feature_aug.size(0), -1)

        feature_aug = self.bottleneck_layer(feature_aug)
        # output1 = self.fc(feature_aug)
        # softmax_outputs1 = F.softmax(output1, dim=1)
        # feat_oa = torch.matmul(softmax_outputs1, self.centroid)

        features = features_p + feature_aug

        # CDA
        outputs2 = self.fc2(features)
        softmax_outputs = F.softmax(outputs2, dim=1)
        return features, outputs2, softmax_outputs, pre_output, features_p

    def norm_adj_matrix(self,edge):
        D_new_hat = np.array(np.sum(edge, axis=0))
        D_new_inv = np.power(D_new_hat, -0.5)
        D_new_mat = np.matrix(np.diag(D_new_inv))
        norm_adj_new = np.asmatrix(D_new_mat) @ np.asmatrix(edge)
        norm_adj_new = norm_adj_new @ np.asmatrix(D_new_mat)
        norm_adj_new = torch.from_numpy(norm_adj_new).float().cuda()
        return norm_adj_new

    def pagerank_power_batch(self, adj_mat, personalize, p=0.85, max_iter=100, tol=1e-8):

        batch_size = personalize.size(0)
        n = adj_mat.size(-1)

        # 对每行逐元素相除
        tmp = torch.sum(adj_mat, dim=1)
        transition_matrix = adj_mat / tmp
        W = p * transition_matrix

        s = personalize * n
        z_T = torch.ones((1, n)) * (1 - p) / n
        W = W.unsqueeze(0).expand(batch_size, n, n)
        z_T = z_T.unsqueeze(0).expand(batch_size, 1, n).view(batch_size, -1, n).cuda()

        x = s
        old_x = torch.zeros((batch_size, n, 1)).cuda()

        for i in range(max_iter):
            old_x = x
            tmp = torch.bmm(z_T, x)
            x = torch.bmm(W, x) + torch.bmm(s, tmp)

        x = F.normalize(x, dim=1)
        x = (personalize + x)
        x = x.view(batch_size, -1)
        return x

class model(object):
    def __init__(self, cfg, use_gpu=True, symmetric_graph=None, node_num=87):
        self.base_net = Resnet(cfg.backbone)
        self.classifier = Classifier(self.base_net.output_num, cfg, symmetric_graph=symmetric_graph, node_num=node_num)

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
        self.prototype_feat_s = torch.rand(cfg.share_class_num, self.bottleneck_dim).cuda()
        mask = [1 if i < cfg.share_class_num else 0 for i in range(cfg.class_num)]
        self.mask = torch.ByteTensor(mask).cuda()
    # init MIM
    def init_mom_softmax(self, train_loader_len):
        self.momentum_softmax_target = loss.MomentumSoftmax(
            self.class_num, m=train_loader_len)

    def get_loss(self, inputs, labels_source):
        bs = labels_source.size(0) // 2

        features_ = self.base_net(inputs)
        features, outputs2, _, pre_output, features_p = self.classifier(features_)
        feature_source_share = features.narrow(0, 0, bs)
        outputs_source = pre_output.narrow(0, 0, bs * 2)
        outputs_source_share = outputs2.narrow(0, 0, bs)
        labels_source_share = labels_source.narrow(0, 0, bs)

        feature_target = features.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))
        outputs_target = outputs2.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))
        feature_source4center = features_p.narrow(0, 0, bs*2)
        up_center_s = update_center(feature_source4center, labels_source, self.classifier.centroid, self.class_num)
        self.classifier.centroid = up_center_s.detach()

        # update prototype_feat_s
        up_feat_s = update_prototype(feature_source_share, labels_source_share, self.prototype_feat_s)
        self.prototype_feat_s = up_feat_s.detach()

        # label smooth regular loss
        src_ = loss.CrossEntropyLabelSmooth(reduction='none', num_classes=self.class_num, epsilon=self.smooth)(
            outputs_source_share, labels_source_share)
        weight_src = self.class_weight_src[labels_source_share].unsqueeze(0)
        classifier_loss = torch.sum(weight_src * src_) / (torch.sum(weight_src).item())
        # classifier_loss += nn.CrossEntropyLoss()(outputs_source, labels_share)
        src_ = loss.CrossEntropyLabelSmooth(reduction='none', num_classes=self.class_num, epsilon=self.smooth)(
            outputs_source, labels_source)
        weight_src = self.class_weight_src[labels_source].unsqueeze(0)
        classifier_loss +=  torch.sum(weight_src * src_) / (torch.sum(weight_src).item())

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
        _, _, softmax_outputs, _, _ = self.classifier(features)
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