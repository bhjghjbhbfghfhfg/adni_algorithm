# -*- coding: utf-8 -*-

import time
import torch
import math
import numpy as np
import os
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import multi_model
import vgg_3d
from config_multi import get_args_multi
from dataloader_multi import DataSet, load_data_multi
import gc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report
import pytorch_warmup as warmup
import torch.optim as optim
from conv4d_multi import resnet10
from torch.utils.tensorboard import SummaryWriter

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
        import ipdb;
        ipdb.set_trace()
        # 这里的data可以自定义取
        # 这里获得的结果是未进行归一化之前的概率，但是进行softmax之前必须要进行归一化。而Entropy里面自带归一化操作。
        output = ResNet_3D(data1, data2)
        loss = LabelSmoothingCrossEntropy()(output, label)

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

    return mean_loss, source_correct, total_label


def val_model(epoch, val_data, ResNet_3D, log_best):
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

    with torch.no_grad():
        for (data1, data2, labels) in tqdm(val_data, total=n_batches):
            data1 = data1.cuda()
            data2 = data2.cuda()
            labels = labels.cuda()

            output = ResNet_3D(data1, data2)
            total_loss = F.cross_entropy(output, labels)

            _, predicted = torch.max(output, 1)
            # source_correct += predicted.eq(labels.data.view_as(predicted)).cpu().sum()
            # total_label += labels.size(0)

            true_label.extend(list(labels.cpu().flatten().numpy()))
            data_pre.extend(list(predicted.cpu().flatten().numpy()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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
    log_best.write(log_str)
    del correct, total, true_label, data_pre, data1, data2, labels, output, TN, FP, FN, TP
    gc.collect()
    return ACC, SEN, SPE, AUC, mean_loss


def test_model(test_data, ResNet_3D):
    # 注意为了排除BN和Dropout对测试影响
    ResNet_3D.eval()
    print('Test a model on the test data...')
    correct = 0
    total = 0
    true_label = []
    data_pre = []
    # 迭代次数
    n_batches = len(test_data)

    with torch.no_grad():
        for (data1, data2, labels) in tqdm(test_data, total=n_batches):
            data1 = data1.cuda()
            data2 = data2.cuda()
            labels = labels.cuda()

            output = ResNet_3D(data1, data2)

            _, predicted = torch.max(output, 1)
            true_label.extend(list(labels.cpu().flatten().numpy()))
            data_pre.extend(list(predicted.cpu().flatten().numpy()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # res = classification_report(true_label, data_pre)
    # print(res)
    TN, FP, FN, TP = confusion_matrix(true_label, data_pre).ravel()
    ACC = 100 * (TP + TN) / (TP + TN + FP + FN)
    SEN = 100 * (TP) / (TP + FN)
    SPE = 100 * (TN) / (TN + FP)
    AUC = 100 * roc_auc_score(true_label, data_pre)
    print('The result of test data: \n')
    print('TP:', TP, 'FP:', FP, 'FN:', FN, 'TN:', TN)
    print('ACC: %.4f %%' % ACC)
    print('SEN: %.4f %%' % SEN)
    print('SPE: %.4f %%' % SPE)
    print('AUC: %.4f %%' % AUC)
    del correct, total, true_label, data_pre, data1, data2, labels, output, TN, FP, FN, TP
    gc.collect()
    return ACC, SEN, SPE, AUC


if __name__ == '__main__':
    args = get_args_multi()
    model_save_path = './model/3DResNet_conv' + args.gpu + '.pt'
    print(vars(args))
    # SEED = args.seed
    # np.random.seed(SEED) # 如果 SEED 相同，每次生成的随机数相同
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    writer = SummaryWriter("log/" + "two_class_conv4d" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + str(args.lr))

    train_data = load_data_multi(args, args.train_root_path, args.AD_dir, args.CN_dir, train=True)
    val_data = load_data_multi(args, args.val_root_path, args.AD_dir, args.CN_dir)
    test_data = load_data_multi(args, args.test_root_path, args.AD_dir, args.CN_dir)

    ResNet_3D = resnet10().cuda()

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

    for epoch in range(1, args.nepoch + 1):
        # 训练模型
        fo = open("test.txt", "a")
        log_best = open('log_best.txt', 'a')

        train_loss, train_correct, len_train = train_epoch(epoch, ResNet_3D, train_data, fo, optimizer, scheduler)
        # 打印当前训练损失、最小损失和准确率
        if train_loss < train_best_loss:
            train_best_loss = train_loss
        train_acc = 100. * train_correct / len_train
        # tensorboard 打印
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_acc", train_acc, epoch)

        if train_acc > train_best_acc:
            max_acc = train_acc
        print('current loss: ', train_loss, 'the best loss: ', train_best_loss)
        print(f'train_correct/train_data: {train_correct}/{len_train} accuracy: {train_acc:.2f}%')

        # 模型用于验证集并将评价指标写入txt
        ACC, SEN, SPE, AUC, test_loss = val_model(epoch, val_data, ResNet_3D, log_best)
        if ACC > val_best_acc:
            val_best_acc = ACC
            t_SEN = SEN
            t_SPE = SPE
            t_AUC = AUC
            # 保存模型
            print('model saved...')
            torch.save(ResNet_3D.state_dict(), model_save_path)
        log_best.write('The best result:\n')
        log_best.write('ACC:  ' + str(val_best_acc) + '  SEN:  ' + str(t_SEN) + '  SPE:  ' + str(
            t_SPE) + '  AUC:  ' + str(t_AUC) + '\n\n')

        # tensorboard 打印
        writer.add_scalar("val_loss", test_loss, epoch)
        writer.add_scalar("val_acc", ACC, epoch)

        print(f'The train acc of this epoch: {train_acc:.2f}%')
        print(f'The best val acc: {val_best_acc:.2f}% \n')
        print(f'The val acc: {ACC:.2f}%')
        print(f'The val loss: {test_loss} \n')
        fo.write('train_acc: '+str(train_acc)+' The current total loss: '+str(train_loss)+' The best loss: '+str(train_best_loss)+'\n\n')

        # 保存训练结果用于画图
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc.cpu().item())
        test_loss_all.append(test_loss.cpu().item())
        test_acc_all.append(ACC)

        del train_loss, train_correct, len_train, train_acc
        gc.collect()
        fo.close()
        log_best.close()

    # 将模型用于测试集
    ResNet_3D.load_state_dict(torch.load(model_save_path))
    test_ACC, test_SEN, test_SPE, test_AUC = test_model(test_data, ResNet_3D)

    writer.add_scalar("test_ACC", test_ACC, 1)
    writer.add_scalar("test_SEN", test_SEN, 1)
    writer.add_scalar("test_SPE", test_SPE, 1)
    writer.add_scalar("test_AUC", test_AUC, 1)
    time_use = time.time() - since
    print("Train and Test complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    writer.close()
    train_process = pd.DataFrame(
        data={"epoch": range(args.nepoch),
              "train_loss_all": train_loss_all,
              "train_acc_all": train_acc_all,
              "test_loss_all": test_loss_all,
              "test_acc_all": test_acc_all}
    )
    train_process.to_csv('train_process_result.csv')

    # 画图
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.train_loss_all, label="Train loss")
    plt.plot(train_process.epoch, train_process.test_loss_all, label="Test loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch, train_process.train_acc_all, label="Train acc")
    plt.plot(train_process.epoch, train_process.test_acc_all, label="Test acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()

    plt.savefig("accuracy_loss_lr.png")
    # plt.show()


