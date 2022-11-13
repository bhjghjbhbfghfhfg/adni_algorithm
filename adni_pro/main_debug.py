# -*- coding: utf-8 -*-

import time
import torch
import math
import numpy as np
import os
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import multi_model
import vgg_3d
from resnet_3d import get_net
from config_multi import get_args_multi
from dataloader_multi_patient_addition import DataSet, load_data_multi
import gc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report
import pytorch_warmup as warmup
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

def linear_combination(x, y, epsilon):
    return epsilon*x + (1-epsilon)*y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.2, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)

def train_epoch(epoch, ResNet_3D, train_data, fo, optimizer, scheduler):
    ResNet_3D.train() # 这个标识意味着进入训练标志
    # 迭代次数
    n_batches = len(train_data)

    total_loss = 0
    source_correct = 0
    total_label = 0

    for (data1, data2, label) in tqdm(train_data, total=n_batches):

        data1, data2, label = data1.cuda(), data2.cuda(), label.cuda()
        # 这里的data可以自定义取
        # 这里获得的结果是未进行归一化之前的概率，但是进行softmax之前必须要进行归一化。而Entropy里面自带归一化操作。
        output, output_mri, output_pet = ResNet_3D(data1, data2)
        loss_multi = LabelSmoothingCrossEntropy()(output, label)
        loss_mri = LabelSmoothingCrossEntropy()(output_mri, label)
        loss_pet = LabelSmoothingCrossEntropy()(output_pet, label)
        loss = loss_multi + loss_mri + loss_pet

        cls_multi = softmax(output)
        cls_mri = softmax(output_mri)
        cls_pet = softmax(output_pet)

        output = 1 * cls_multi + 0 * cls_mri + 1 * cls_pet

        _, preds = torch.max(output, 1)

        source_correct += preds.eq(label.data.view_as(preds)).cpu().sum()
        total_label += label.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    lr_current = scheduler.get_last_lr()
    scheduler.step()
    acc = 100. * source_correct / total_label
    mean_loss = total_loss / n_batches

    print(f'Epoch: [{epoch:02d}], Loss: {mean_loss:.6f}')

    log_str = 'Epoch: '+str(epoch)\
              +' Loss: '+str(mean_loss)\
              +' train_acc: '+str(acc)+'\n'
    fo.write(log_str)
    del acc, train_data, n_batches
    gc.collect()

    return mean_loss, source_correct, total_label, lr_current


def val_model(epoch, val_data, ResNet_3D, log_best, state, writer):
    # 注意为了排除BN和Dropout对测试影响
    ResNet_3D.eval()
    print('Test a model on the val data...')
    correct = 0
    total = 0
    total_loss = 0
    true_label = []
    data_pre = []
    # 迭代次数
    n_batches = len(val_data)

    # source_correct = 0
    # total_label = 0

    mri_pre = []
    pet_pre = []
    multi_pre = []
    final_pre = []
    weight = []

    all_features = torch.tensor([], device=0)
    all_features_mri = torch.tensor([], device=0)
    all_features_pet = torch.tensor([], device=0)
    all_labels = torch.tensor([], device=0)

    multi = []
    mri = []
    pet = []

    index = 0
    with torch.no_grad():
        for (data1, data2, labels) in tqdm(val_data, total=n_batches):
            data1 = data1.cuda()
            data2 = data2.cuda()
            labels = labels.cuda()

            output, output_mri, output_pet = ResNet_3D(data1, data2)

            if epoch == -1:
                # 测试集准备进行 tensorboard embedding
                all_features = torch.cat((all_features, output), dim=0)
                all_features_mri = torch.cat((all_features_mri, output_mri), dim=0)
                all_features_pet = torch.cat((all_features_pet, output_pet), dim=0)
                all_labels = torch.cat((all_labels, labels), dim=0)

                for i in range(get_args_multi().batch_size):
                    multi.append(str(4 * index + i) + 'multi')
                    mri.append(str(4 * index + i) + 'mri')
                    pet.append(str(4 * index + i) + 'pet')
                index = index + 1

            if state == 0:
                loss_mri = F.cross_entropy(output_mri, labels)
                loss = loss_mri
            elif state == 1:
                loss_pet = F.cross_entropy(output_pet, labels)
                loss = loss_pet
            elif state == 2:
                loss_multi = F.cross_entropy(output, labels)
                loss_mri = F.cross_entropy(output_mri, labels)
                loss_pet = F.cross_entropy(output_pet, labels)
                loss = loss_multi  # + loss_mri + loss_pet
            total_loss += loss.item()

            if args.state == 0:
                cls_mri = softmax(output_mri)
                output = cls_mri
            elif args.state == 1:
                cls_pet = softmax(output_pet)
                output = cls_pet
            elif args.state == 2:
                cls_multi = softmax(output)
                cls_mri = softmax(output_mri)
                cls_pet = softmax(output_pet)
                output = 0 * cls_multi + 0 * cls_mri + 1 * cls_pet

            _, predicted = torch.max(output, 1)
            if args.state == 2:
                _, predicted_mri = torch.max(output_mri, 1)
                _, predicted_pet = torch.max(output_pet, 1)
                predicted_labels = labels

                mri_pre += predicted.cpu().flatten().tolist()  # mri AD人数
                pet_pre += predicted_mri.cpu().flatten().tolist()  # pet AD人数
                multi_pre += predicted_pet.cpu().flatten().tolist()  # multi AD人数
                final_pre += predicted_labels.cpu().flatten().tolist()  # labels AD人数
                # weight += pred.cpu().tolist()  # attention关注的权重
                # import pdb; pdb.set_trace()
                # for x in range(0, args.batch_size):
                #     writer.add_scalar("weight", pred.cpu().tolist()[x][0], index * args.batch_size + x)
                index = index + 1
            # source_correct += predicted.eq(labels.data.view_as(predicted)).cpu().sum()
            # total_label += labels.size(0)

            true_label.extend(list(labels.cpu().flatten().numpy()))
            data_pre.extend(list(predicted.cpu().flatten().numpy()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 画图embedding
    if epoch == -1:
        res = []
        for x in all_labels:
            res.append(x.item())
        # import pdb;
        # pdb.set_trace()
        writer[0].add_embedding(all_features, res)
        writer[1].add_embedding(all_features_mri, res)
        writer[2].add_embedding(all_features_pet, res)

    mean_loss = total_loss / n_batches
    TN, FP, FN, TP = confusion_matrix(true_label, data_pre).ravel()
    ACC = 100 * (TP + TN) / (TP + TN + FP + FN)
    SEN = 100 * (TP) / (TP + FN)
    SPE = 100 * (TN) / (TN + FP)
    AUC = 100 * roc_auc_score(true_label, data_pre)
    print('TP:', TP, 'FP:', FP, 'FN:', FN, 'TN:', TN)
    print('ACC: %.4f %%' % ACC)
    print('SEN: %.4f %%' % SEN)
    print('SPE: %.4f %%' % SPE)
    print('AUC: %.4f %%' % AUC)
    log_str = 'Epoch: ' + str(epoch) \
              + '\n' \
              + 'TP: ' + str(TP) + ' TN: ' + str(TN) + ' FP: ' + str(FP) + ' FN: ' + str(FN) \
              + '  ACC:  ' + str(ACC) \
              + '  SEN:  ' + str(SEN) \
              + '  SPE:  ' + str(SPE) \
              + '  AUC:  ' + str(AUC) \
              + '\n'
    #log_best.write(log_str)

    people = pd.DataFrame(
        data={# "epoch": range(args.nepoch),
              "mri_pre": mri_pre,
              "pet_pre": pet_pre,
              "multi_pre": multi_pre,
              "final_pre": final_pre}
              # "weight": weight}
    )

    people.to_csv('pre_data_attention.csv')

    del correct, total, true_label, data_pre, data1, data2, labels, output, TN, FP, FN, TP
    gc.collect()
    return ACC, SEN, SPE, AUC, mean_loss

def softmax(inx):
    return nn.Softmax(-1)(inx)

if __name__ == '__main__':
    args = get_args_multi()
    print(vars(args))
    SEED = args.seed
    np.random.seed(SEED) # 如果 SEED 相同，每次生成的随机数相同
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    writer_multi = SummaryWriter("log/" + 'multi')
    writer_mri = SummaryWriter("log/" + 'mri')
    writer_pet = SummaryWriter("log/" + 'pet')

    train_data = load_data_multi(args, args.train_root_path, args.AD_dir, args.CN_dir, train=True)
    val_data = load_data_multi(args, args.val_root_path, args.AD_dir, args.CN_dir)
    test_data = load_data_multi(args, args.test_root_path, args.AD_dir, args.CN_dir)

    ResNet_3D_res1 = get_net('ResNet10').cuda()
    # ResNet_3D_res1 = get_net('ResNet10')
    ResNet_3D_res2 = get_net('ResNet10').cuda()
    # ResNet_3D_res2 = get_net('ResNet10')
    # 略显误导，但是方便后续修改，ResNet_3D 这个是融合后的模型
    # 这个是特征融合
    # ResNet_3D = multi_model.MultiModel(ResNet_3D_res1, ResNet_3D_res2).cuda()
    # 这个是双线性池化
    # ResNet_3D = multi_model.MultiModelBP(ResNet_3D_res1, ResNet_3D_res2).cuda()
    # ResNet_3D = multi_model.MultiModel(ResNet_3D_res1, ResNet_3D_res2)
    # 这里使用attentino
    ResNet_3D = multi_model.MultiModelAttention(ResNet_3D_res1, ResNet_3D_res2).cuda()

    train_best_loss = 10000
    train_best_acc = 0
    val_best_acc = 0
    t_SEN = 0
    t_SPE = 0
    t_AUC = 0
    t_precision = 0
    t_f1 = 0

    train_loss_all = []
    train_acc_all = []
    test_loss_all = []
    test_acc_all = []
    LEARNING_RATE_all = []

    since = time.time()

    t = 10  # warmup
    T = args.nepoch
    n_t = 0.5
    lr_init = get_args_multi().lr
    lambda1 = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t else 0.1 if n_t * (
            1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.1 else n_t * (
            1 + math.cos(math.pi * (epoch - t) / (T - t)))
    optimizer = torch.optim.Adam(params=ResNet_3D.parameters(), lr=lr_init, weight_decay=5e-4)
    print(sum([np.prod(p.shape) for p in ResNet_3D.parameters()]))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # 将模型用于测试集
    # ResNet_3D.load_state_dict(torch.load('./model/3DResNet.pt'))
    state_dict = torch.load('./model/3DResNet.pt')
    state_dict = {k: v for k, v in state_dict.items()
                  if not k in ['classify.weight', 'proj_1.weight', 'proj_1.bias', 'proj_2.weight', 'proj_2.bias']}
    ResNet_3D.load_state_dict(state_dict, strict=False)
    val_model(-1, test_data, ResNet_3D, None, state=args.state, writer=[writer_multi, writer_mri, writer_pet])

    time_use = time.time() - since
    print("Train and Test complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    # train_process = pd.DataFrame(
    #     data={"epoch": range(args.nepoch),
    #           "train_loss_all": train_loss_all,
    #           "train_acc_all": train_acc_all,
    #           "test_loss_all": test_loss_all,
    #           "test_acc_all": test_acc_all}
    # )
    # train_process.to_csv('train_process_result.csv')
    writer_multi.close()
    writer_mri.close()
    writer_pet.close()

    # 画图
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(train_process.epoch, train_process.train_loss_all, label="Train loss")
    # plt.plot(train_process.epoch, train_process.test_loss_all, label="Test loss")
    # plt.legend()
    # plt.xlabel("epoch")
    # plt.ylabel("Loss")
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(train_process.epoch, train_process.train_acc_all, label="Train acc")
    # plt.plot(train_process.epoch, train_process.test_acc_all, label="Test acc")
    # plt.legend()
    # plt.xlabel("epoch")
    # plt.ylabel("acc")
    # plt.legend()

    # plt.subplot(1, 3, 3)
    # plt.plot(train_process.epoch, train_process.LEARNING_RATE_all, label="LEARNING_RATE_all")
    # plt.legend()
    # plt.xlabel("epoch")
    # plt.ylabel("LEARNING_RATE")
    # plt.legend()
    # plt.savefig("accuracy_loss_lr.png")
    # plt.show()


