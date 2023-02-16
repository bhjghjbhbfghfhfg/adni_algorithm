import os
import shutil
import pandas as pd  # 导入pandas库

## 主要是将下载的数据根绝csv的标签划分为 AD CN MCI三类，目前是所有的数据都放到train里面了

# path_origin = '/Users/changxinge/新下载的数据位置/multi_model/PET/AD'
# path_origin = '/Users/changxinge/新下载的数据位置/multi_model/PET/CN'
# path_origin = '/Users/changxinge/新下载的数据位置/multi_model/PET/MCI'
# path_origin = '/Users/changxinge/Desktop/最晚的病人图像/PET/AD'
# path_origin = '/Users/changxinge/Desktop/最晚的病人图像/PET/CN'
# path_origin = '/Users/changxinge/Desktop/最晚的病人图像/PET/MCI'
path_origin = ''
# path_save = '/Users/changxinge/Desktop/最晚的病人图像/PET/acpc_data/AD'
# path_save = '/Users/changxinge/Desktop/最晚的病人图像/PET/acpc_data/CN'
# path_save = '/Users/changxinge/Desktop/最晚的病人图像/PET/acpc_data/MCI'
path_save = ''

# path_list = os.listdir(path_origin)
# path_res = '/Users/changxinge/PycharmProjects/graduate_design/adni_pro/data/mri/train/'
#
# path_label = '/Users/changxinge/PycharmProjects/dataset_graduate/MRI/1022_mri_colin27_pre_10_22_2022.csv'
#
# count = 0
#
# allfiles = []

def getAllFiles(path):
    childFiles = os.listdir(path)
    # print(childFiles)
    for file in childFiles:
        filepath = os.path.join(path, file)
        if os.path.isdir(filepath):
            getAllFiles(filepath)
        elif os.path.isfile(filepath):
            part = os.path.splitext(file)
            if part[1] == ".nii":
                shutil.copy(filepath, path_save)


if __name__ == '__main__':
    getAllFiles(path_origin)

# def find_label(path, image_id):
#     excel_file = path  # 导入excel数据
#     data = pd.read_csv(excel_file, index_col='Image Data ID')
#     # 这个的index_col就是index，可以选择任意字段作为索引index，读入数据
#     print(data.loc[image_id])
#     p = data.loc[image_id]
#     res = p['Group']
#     return res
