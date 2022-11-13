import gc
import nibabel
import numpy
from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np

class DataSet(Dataset):
    def __init__(self, root_path, dir):
        self.root_path = root_path
        self.dir = dir
        self.patient_path = os.path.join(self.root_path, self.dir)
        self.patient = os.listdir(self.patient_path)  # 把路径下的所有文件放在一个列表中
        self.images = []
        for i in range(0, len(self.patient)):
            patientStr = self.patient[i]
            patientURL = os.path.join(self.patient_path, patientStr)
            # 把路径下的所有文件放在一个列表中
            self.images.append(os.listdir(patientURL))

    def __getitem__(self, index):
        label = 0
        image_index1 = self.images[index][0]  # 根据索引获取数据名称
        image_index2 = self.images[index][1]  # 根据索引获取数据名称
        img_path1 = os.path.join(self.patient_path, self.patient[index], image_index1)  # 获取数据的路径或目录
        img_path2 = os.path.join(self.patient_path, self.patient[index], image_index2)  # 获取数据的路径或目录
        if img_path1.find('Uniform') == 1 and img_path1.find('Resolution') == 1:
            tmp = img_path1
            img_path1 = img_path2
            img_path2 = tmp
        img1 = nibabel.load(img_path1).get_fdata()  # 读取数据
        img2 = nibabel.load(img_path2).get_fdata()  # 读取数据
        img1 = img1.astype('float32')
        img2 = img2.astype('float32')
        normalization = 'minmax'
        if normalization == 'minmax':
            # 最小最大归一化
            img_max1 = img1.max()
            img_max2 = img2.max()
            img1 = img1 / img_max1
            img2 = img2 / img_max2
        elif normalization == 'median':
            # 除中位数
            img_fla1 = np.array(img1).flatten()
            index1 = np.argwhere(img_fla1 == 0)
            img_median1 = np.median(np.delete(img_fla1, index1))
            img1 = img1 / img_median1

            img_fla2 = np.array(img2).flatten()
            index2 = np.argwhere(img_fla2 == 0)
            img_median2 = np.median(np.delete(img_fla2, index2))
            img2 = img2 / img_median2

        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        # pet 特殊添加的
        # img = numpy.squeeze(img, axis=4)
        # 根据目录名称获取图像标签（AD或CN）
        if self.dir == 'AD/':
            label = label+1
        elif self.dir == 'CN/':
            label = label
        elif self.dir == 'MCI/':
            label = label+2
        if normalization == 'minmax':
            del img_max1
            del img_max2
        else:
            del img_fla1, index1, img_median1
            del img_fla2, index2, img_median2
        gc.collect()
        return img1, img2, label

    def __len__(self):
        return len(self.images)

def load_data_multi(args, root_path, AD_dir, CN_dir, train=False):
    AD_data = DataSet(root_path, AD_dir)
    CN_data = DataSet(root_path, CN_dir)
    all_Dataset = AD_data + CN_data
    all_loader = DataLoader(all_Dataset, batch_size=args.batch_size, shuffle=train, num_workers=16)
    del all_Dataset
    gc.collect()
    return all_loader



# args = get_args()
# train_data, test_data = load_data(args)
# for step, (b_x, b_y) in enumerate(train_data):
#     if step > 1:
#         break
#
# print(b_x.shape)
# print(b_y.shape)
# print(b_x)
# print(b_y)