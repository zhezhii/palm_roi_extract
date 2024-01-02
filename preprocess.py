#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : 2021
# @Time : 2023/12/31 23:59
# @Software:preprocess.py


import os
import cv2
import numpy as np
from palm_roi_extract_ import evaluate_pic_quality
from palm_roi_extract_ import palm_roi_extract
from gabor_enhance import get_gabor
from gabor_enhance import auto_threshold

# # 获取当前工作目录
# cwd = os.getcwd()
#
# # 定义包含图像的文件夹的路径
# path = os.path.join(cwd, './base')
path = './data/base'

save_path = './data'

# 创建训练文件夹
os.makedirs(os.path.join(save_path, 'train'), exist_ok=True)
# 创建测试文件夹
os.makedirs(os.path.join(save_path, 'test'), exist_ok=True)

# path_ = path + '/1'

for folder in os.listdir(path):
    path_ = os.path.join(path, folder)
    # 循环遍历文件夹中的所有文件夹
    for filefolder in os.listdir(path_):
        # 创建分类文件夹
        train_path = os.path.join(save_path, 'train', filefolder)
        os.makedirs(train_path, exist_ok=True)
        test_path = os.path.join(save_path, 'test', filefolder)
        os.makedirs(test_path, exist_ok=True)

        # 获取文件夹中的文件
        path__ = os.path.join(path_, filefolder)

        """
        优先获取全部的图片 然后再将图片分发给训练集或测试集
        否则容易造成测试集可能没有对应的测试文件
        从而导致无法正确运行代码
        """
        # 创建存储图片列表
        img_ls = []
        # 创建存储图片名字的列表
        img_id_ls = []
        for filename in os.listdir(path__):
            # 检查文件是否为图像
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.bmp'):
                # 加载图像
                img = cv2.imread(os.path.join(path__, filename), 0)
                # 获取ROI区域
                rotated_img, roi, mask = palm_roi_extract(img, filename)
                if mask is None:
                    continue
                # # 图像纹理增强
                enhance = cv2.equalizeHist(roi)
                if cv2.countNonZero(enhance) == 0:
                    continue
                gabor = get_gabor(enhance)
                # 将图像矫正为(192, 192)
                gabor = cv2.resize(gabor, (192, 192))
                auto = auto_threshold(gabor)

                # cv2.imshow('roi', gabor)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # 将获得的图像存储进ls
                img_ls.append(auto)
                # 将图片名字存入ls
                img_id_ls.append(filename)

        # 将图片分发到训练集和测试集
        test_ind = np.random.randint(0, len(img_ls) - 1, 2)
        # 将图片分发到对应数据集
        for ind in range(len(img_ls)):
            fname = folder + img_id_ls[ind].split('.')[0] + '_gabor.jpg'
            if ind in list(test_ind):
                cv2.imwrite(os.path.join(test_path, fname), img_ls[ind])
            else:
                cv2.imwrite(os.path.join(train_path, fname), img_ls[ind])
