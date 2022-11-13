import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._C import Size
from torch.autograd import Variable
from config_multi import get_args_multi

class MultiModel(nn.Module):
    def __init__(
        self,
        res1,
        res2,
        num_classes=get_args_multi().class_num, #分类数
    ):
        super(MultiModel, self).__init__()
        self.res1 = res1
        self.res2 = res2
        self.num_classes = num_classes
        # 这里引入了辅助loss MRI
        self.classify = nn.Linear(1024, self.num_classes, bias=False)
        self.classify_mri = nn.Linear(512, self.num_classes, bias=False)
        self.classify_pet = nn.Linear(512, self.num_classes, bias=False)

    # 特征融合方法forward
    def forward(self, x1, x2):
        res1 = self.res1(x1)
        res2 = self.res2(x2)
        res = torch.cat([res1, res2], -1)
        cls = self.classify(res)
        cls_mri = self.classify_mri(res1)
        cls_pet = self.classify_pet(res2)
        return cls, cls_mri, cls_pet


class MultiModelBP(nn.Module):
    def __init__(
            self,
            res1,
            res2,
            num_classes=get_args_multi().class_num,  # 分类数
    ):
        super(MultiModelBP, self).__init__()
        self.res1 = res1
        self.res2 = res2
        self.num_classes = num_classes
        # 这里引入了辅助loss MRI, 辅助loss PET
        self.classify = nn.Linear(1000, self.num_classes, bias=False)
        self.classify_mri = nn.Linear(512, self.num_classes, bias=False)
        self.classify_pet = nn.Linear(512, self.num_classes, bias=False)

        if get_args_multi().state==2:
            for p in self.res1.parameters():
                p.requires_grad = False
            for p in self.res2.parameters():
                p.requires_grad = False
            for p in self.classify_mri.parameters():
                p.requires_grad = False
            for p in self.classify_pet.parameters():
                p.requires_grad = False

        self.k = 4
        self.proj_1 = nn.Linear(512, 1000 * self.k)
        self.proj_2 = nn.Linear(512, 1000 * self.k)
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.AvgPool1d(self.k, stride=self.k)
        self.state = get_args_multi().state

    # 双线性池化
    def forward(self, x1, x2):
        if self.state == 0 or self.state == 2:
            res1 = self.res1(x1)
            cls_mri = self.classify_mri(res1)
        if self.state == 1 or self.state == 2:
            res2 = self.res2(x2)
            cls_pet = self.classify_pet(res2)
        if self.state == 0:
            return None, cls_mri, None
        if self.state == 1:
            return None, None, cls_pet

        h1 = self.proj_1(res1)[:, None, :]
        h2 = self.proj_2(res2)[:, None, :]

        exp_out = h1 * h2
        exp_out = self.dropout(exp_out)
        z = self.pool(exp_out) * self.k
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        res = F.normalize(z[:, 0])

        cls = self.classify(res)

        return cls, cls_mri, cls_pet


class MultiModelAttention(nn.Module):
    def __init__(
            self,
            res1,
            res2,
            num_classes=get_args_multi().class_num,  # 分类数
    ):
        super(MultiModelAttention, self).__init__()
        self.res1 = res1
        self.res2 = res2
        self.num_classes = num_classes
        # 这里引入了辅助loss MRI, 辅助loss PET
        self.classify = nn.Linear(1024, self.num_classes, bias=False)
        self.classify_all = nn.Linear(1024, 2, bias=False)
        self.classify_mri = nn.Linear(512, self.num_classes, bias=False)
        self.classify_pet = nn.Linear(512, self.num_classes, bias=False)

        if get_args_multi().state==2:
            for p in self.res1.parameters():
                p.requires_grad = False
            for p in self.res2.parameters():
                p.requires_grad = False
            for p in self.classify_mri.parameters():
                p.requires_grad = False
            for p in self.classify_pet.parameters():
                p.requires_grad = False


        self.state = get_args_multi().state

    # attention
    def forward(self, x1, x2):
        if self.state == 0 or self.state == 2:
            res1 = self.res1(x1)
            cls_mri = self.classify_mri(res1)
        if self.state == 1 or self.state == 2:
            res2 = self.res2(x2)
            cls_pet = self.classify_pet(res2)
        if self.state == 0:
            return None, cls_mri, None
        if self.state == 1:
            return None, None, cls_pet

        # import pdb;
        # pdb.set_trace()
        res = torch.cat((res1, res2), dim=1)
        output = self.classify_all(res)
        pred = nn.Softmax(dim=-1)(output)
        # 这里 pre[:,0]是一维度向量，加上None自动变成二纬度向量 eg：b -> b * 1
        output = torch.cat((pred[:,0,None] * res1, pred[:,1,None] * res2), dim=1)
        cls = self.classify(output)
        return cls, cls_mri, cls_pet

class MultiModelAttention_VGG(nn.Module):
    def __init__(
            self,
            res1,
            res2,
            num_classes=get_args_multi().class_num,  # 分类数
    ):
        super(MultiModelAttention_VGG, self).__init__()
        self.res1 = res1
        self.res2 = res2
        self.num_classes = num_classes
        # 这里引入了辅助loss MRI, 辅助loss PET
        self.classify = nn.Linear(256, self.num_classes, bias=False)
        self.classify_all = nn.Linear(256, 2, bias=False)
        self.classify_mri = nn.Linear(128, self.num_classes, bias=False)
        self.classify_pet = nn.Linear(128, self.num_classes, bias=False)

        # if get_args_multi().state==2:
        #     for p in self.res1.parameters():
        #         p.requires_grad = False
        #     for p in self.res2.parameters():
        #         p.requires_grad = False
        #     for p in self.classify_mri.parameters():
        #         p.requires_grad = False
        #     for p in self.classify_pet.parameters():
        #         p.requires_grad = False


        self.state = get_args_multi().state

    # attention
    def forward(self, x1, x2):
        if self.state == 0 or self.state == 2:
            res1 = self.res1(x1)
            cls_mri = self.classify_mri(res1)
        if self.state == 1 or self.state == 2:
            res2 = self.res2(x2)
            cls_pet = self.classify_pet(res2)
        if self.state == 0:
            return None, cls_mri, None
        if self.state == 1:
            return None, None, cls_pet

        # import pdb;
        # pdb.set_trace()
        res = torch.cat((res1, res2), dim=1)
        output = self.classify_all(res)
        pred = nn.Softmax(dim=-1)(output)
        # 这里 pre[:,0]是一维度向量，加上None自动变成二纬度向量 eg：b -> b * 1
        output = torch.cat((pred[:,0,None] * res1, pred[:,1,None] * res2), dim=1)
        cls = self.classify(output)
        return cls, cls_mri, cls_pet