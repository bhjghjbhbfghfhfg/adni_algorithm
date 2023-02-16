import os
import shutil
import pandas as pd  # 导入pandas库

# 这个函数的目的就是从MRI和PET中找出相同病人的两种图片信息
mri_path = '/Users/changxinge/Desktop/最晚的病人图像/ready_mri_data/MCI/'
pet_path = '/Users/changxinge/Desktop/最晚的病人图像/ready_pet_data/MCI/'

save_path = '/Users/changxinge/Desktop/最晚的病人图像/multi_data_later/MCI/'

def getAllFiles(**kwargs):
    childFiles_mri = os.listdir(mri_path)
    print(childFiles_mri)
    for file_mri in childFiles_mri:
        childFiles_pet = os.listdir(pet_path)
        for file_pet in childFiles_pet:
            patientID_mri = file_mri[11:21]
            patientID_pet = file_pet[11:21]
            if (patientID_mri == patientID_pet):
                patientDIR = os.path.join(save_path, patientID_mri)
                os.makedirs(patientDIR)
                origin_mri_file = mri_path + file_mri
                origin_pet_file = pet_path + file_pet
                shutil.copy(origin_mri_file, patientDIR)
                shutil.copy(origin_pet_file, patientDIR)
                break

if __name__ == '__main__':
    getAllFiles()
