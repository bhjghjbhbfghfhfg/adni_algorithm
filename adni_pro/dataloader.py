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
        self.image_path = os.path.join(self.root_path, self.dir)
        self.images = os.listdir(self.image_path)  # 把路径下的所有文件放在一个列表中

    def __getitem__(self, index):
        label = 0
        image_index = self.images[index]  # 根据索引获取数据名称
        img_path = os.path.join(self.image_path, image_index)  # 获取数据的路径或目录
        img = nibabel.load(img_path).get_fdata()  # 读取数据
        img = img.astype('float32')
        normalization = 'median'
        if normalization == 'minmax':
            # 最小最大归一化
            img_max = img.max()
            img = img / img_max
        elif normalization == 'median':
            # 除中位数
            img_fla = np.array(img).flatten()
            index = np.argwhere(img_fla == 0)
            img_median = np.median(np.delete(img_fla, index))
            img = img / img_median

        img = np.expand_dims(img, axis=0)
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
            del img_max
        else:
            del img_fla, index, img_median
        gc.collect()
        return img, label

    def __len__(self):
        return len(self.images)

def load_data(args, root_path, AD_dir, CN_dir, train=False):
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