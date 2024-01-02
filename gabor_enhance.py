#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : 2021
# @Time : 2024/01/02 16:30
# @Software:gabor_enhance.py

import cv2
import numpy as np
# import matplotlib.pyplot as plt

def build_filters(ksize, sigma, theta, lambd, gamma):
    filters = []
    for i in range(0, 4):
        theta = i * np.pi / 4
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


def get_gabor(img):
    # ksizes = [7, 9, 11, 13, 15, 17]
    # sigmas = [3, 5, 7, 9]
    # res = []
    # for ksize in ksizes:
    #     for sigma in sigmas:
    filters = build_filters(ksize=11, sigma=7, theta=0, lambd=10, gamma=0.5)
    res = process(img, filters)

    # 将图像展示在同一幅图中
    # fig, axs = plt.subplots(nrows=len(ksizes), ncols=len(sigmas), figsize=(12, 16))
    # for i, ax in enumerate(axs.flat):
    #     ax.imshow(res[i])
    # plt.show()
    return res


def auto_threshold(img):
    img_ = img.copy()
    img_ = cv2.medianBlur(img_, 5)
    img_ = cv2.adaptiveThreshold(img_, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return img_