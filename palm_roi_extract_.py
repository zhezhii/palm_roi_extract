#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : 2021
# @Time : 2024/01/01 1:31
# @Software:palm_roi_extract_.py

import cv2
import numpy as np
from scipy.signal import find_peaks


# def white_col_detect(img, limit=0.5, axis=0):
#     """
#     检测图像中的白色列
#     :param img: 图像
#     :param limit: 白色像素占比阈值
#     :return: 是否大于该阈值limit
#     """
#     # 拷贝图像
#     img_ = img.copy()
#     # 统计白色行
#     white_col = np.sum(img_, axis=1 - axis) == 0
#     # 统计白色行数
#     count = len(white_col) - np.count_nonzero(white_col)
#     # 计算白色行占比
#     ratio = count / img_.shape[axis]
#     # 判断是否大于阈值
#     if ratio > limit:
#         return True


def evaluate_pic_quality(img, desp="图像"):
    """
    评估图像质量
    :param img: 图像
    :param desp: 图像路径 作为判别到达是哪个图像质量不够好
    :return: 若图像质量不够好, 则返回None, 否则返回处理后的图像和对应的ROI
    """
    # 拷贝图像
    img_ = img.copy()
    # 高斯滤波
    blur = cv2.GaussianBlur(img_, (5, 5), 0)

    # OTSU二值化
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 膨胀操作
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(thresh, kernel, iterations=1)


    # 轮廓检测
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 最大轮廓效果并不理想, 故尝试直接删去小区域, 取 0.5 ~ 0.95 之间的区域
    """
    所以说 都是这个傻逼轮廓检测的锅
    每次效果垃圾都是因为轮廓检测的问题
    """
    img_area = img_.shape[0] * img_.shape[1]
    contours_ = [cnt for cnt in contours if img_area * 0.95 > cv2.contourArea(cnt) > img_area * 0.3]

    # 若轮廓数目为0, 则认为图像质量太好了, 不是傻逼设备1的图像, 直接走近路
    if len(contours_) == 0:
        # 构造背景
        bg = np.zeros_like(img_)

        # 轮廓检测 仅检测最大轮廓
        max_contour = max(contours[1:], key=cv2.contourArea)

        cv2.fillPoly(bg, [max_contour], (255, 255, 255))

        # # 如果白色区域过多, 则认为图像质量不够好
        # # 如果纯白行数大于图像高度的 0.7, 则认为图像质量不够好
        # if white_col_detect(bg, 0.8):
        #     print(1)
        #     return None, None
        #
        # # 同理, 如果纯白列数大于图像宽度的 0.5, 则认为图像质量不够好
        # if white_col_detect(bg, 0.8, 1):
        #     print(2)
        #     return None, None

        # 按位与 得到原图像在最大轮廓外的部分
        result = cv2.bitwise_and(img_, bg)

        return result, bg

    # 画出轮廓
    bg = np.ones_like(img_) * 255
    cv2.drawContours(bg, contours_, -1, (0, 255, 0), 3)


    # 轮廓检测 仅检测最大轮廓
    contours, hierarchy_ = cv2.findContours(bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours[1:], key=cv2.contourArea)

    """
    由于图像质量问题, 可能会导致轮廓检测成手掌外部, 考虑手掌位于中心部分, 故考察轮廓中心是否位于轮廓内部
    """

    # 计算轮廓中心
    M = cv2.moments(max_contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # 判断轮廓中心是否位于轮廓内部 若不在
    if cv2.pointPolygonTest(max_contour, (cx, cy), False) >= 0:
        # 画出最大轮廓
        mask = np.zeros_like(img_)
        cv2.fillPoly(mask, [max_contour], (255, 255, 255))

        # 按位与 得到原图像在最大轮廓内的部分
        result = cv2.bitwise_and(img_, mask)
    # 否则, 轮廓中心位于轮廓内部,
    else:
        mask = np.ones_like(img_) * 255
        cv2.fillPoly(mask, [max_contour], (0, 0, 0))

        # 按位与 得到原图像在最大轮廓外的部分
        result = cv2.bitwise_and(img_, mask)

    # 重新获得ROI
    # 拷贝图像
    img__ = img_.copy()
    # 高斯滤波
    blur_ = cv2.GaussianBlur(img__, (3, 3), 0)

    # OTSU二值化
    ret, thresh = cv2.threshold(blur_, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = 255 - thresh

    return result, thresh


def rdf_min(img, left, right, pref):
    """
    模拟RDF函数
    :param img: mask图像
    :param left: 左侧边界点
    :param right: 右侧边界点
    :param pref: 参考点
    :return: 两边界点之间轨迹到参考点的距离最小点
    """
    """
    还是得关注于轮廓来求更加简便
    ...
    还是算了 傻逼轮廓 还是检测不正确 说实话 传统CV是真的烂
    
    规定搜索策略:
    从left列到right列, 截止到pref行, 依次遍历所有列, 查看这些行的轮廓处与参考点的距离
    合适的边界应该满足 上方是黑色部分 而下方是白色部分
    傻逼正常轮廓是识别不出来的 非得自己手搓一个效果很差的轮廓搜索策略
    """
    # 拷贝图像
    img_ = img.copy()
    # 参考像素
    white = img_[tuple(left)]
    # 最小距离
    min_dist = np.sqrt(sum(np.square(left - pref)))
    # # 最小距离点
    # min_point = left
    # 上一个点位置
    last = left
    """
    修改策略:
    考察距离变化方式, 如距离变化取到了某个极值, 则认为此处为某个峰谷点
    这是需要被记录的
    算了 没必要精益求精 后面谷点的搜索还会是有很多问题 需要考虑的情况太多了
    反正是小作业, 就这样吧
    """
    # 边界点
    boundary = []

    # # 记录当前距离变化趋势
    # trend = True  # True为上升趋势 False为下降趋势
    # # 记录上一个距离
    # last_dist = min_dist
    # 搜索列
    for col in range(left[1] + 1, right[1]):
        # area = img_[: pref[0], col]
        # for i in range(len(area)):
        #     # 如果此处为黑色而下方以及下方5个像素点为白色 则视为边界点
        #     if area[i] != white and img_[i + 1, col] == white and img_[i + 5, col] == white:
        #         # 计算与参考点的距离
        #         dist = np.sqrt(sum(np.square(np.array([i, col]) - pref)))
        #         if dist < min_dist:
        #             min_dist = dist
        #             min_point = np.array([i, col])
        """
        再度思考一个合适的搜索策略
        考察当前位置右一个单位的像素值, 若为黑色, 则需要向下搜索边界值, 直到找到白色像素值
        同理, 若为白色, 则需要向上搜索边界值, 直到找到黑色像素值
        当然也需要制作一个微型判别器, 以防止搜索到干扰点 # 效果很好 暂时懒得写
        
        设置一个合适的最值提取器:
        首先设置上升和下降两个趋势, 以及上一个距离
        由于需要将当前距离与前一个距离进行比较, 故需要记录上一个点的位置和对应的距离
        1、考察距离变化趋势, 若距离变化趋势为上升 且 当前距离小于上一个距离 则认为上一个点为峰点
        2、若距离变化趋势为下降 且 当前距离大于上一个距离 则认为上一个点为谷点
        
        这里需要重点量化一下变化趋势, 这对搜索结果的影响很大
        优先搜索出全部的边界点
        再对这些轮廓点进行分析, 以确定峰谷点
        """
        # 由于当前位置是左侧边界点, 故向右搜索
        if img_[last[0], col] != white:
            # 向下搜索
            for row in range(last[0] + 1, pref[0]):
                # 若此处为黑色, 则为边界点
                if img_[row, col] == white:
                    # 添加边界点
                    boundary.append(np.array([row, col]))
                    # 结束搜索
                    break
        # 否则, 右侧是白色, 故向上搜索
        else:
            # 向上搜索
            for row in range(last[0] - 1, 0, -1):
                # 若此处为黑色, 则为边界点
                if img_[row, col] != white:
                    # 添加边界点
                    boundary.append(np.array([row, col]))
                    # 结束搜索
                    break

    # 2、已然得到了全部的边界点, 现在需要对这些边界点进行分析, 以确定峰谷点
    # 由于噪声的存在, 可以考虑将边界点进行平滑处理
    # 这里以step个点为一个单位进行平滑处理, 然后再进行分析
    # """
    # 平滑处理:
    # 计算全部的边界点到参考点的距离
    #
    # """
    # step
    step = 3
    # dist存储列表
    dist = []
    # 平滑后的边界点
    boundary_smooth = []
    # 转化为numpy数组
    boundary = np.array(boundary).reshape(-1, 2)
    for i in range(0, boundary.shape[0], step):
        # 计算平均值
        dist.append(np.sqrt(sum(np.square(boundary[i: min(i + step, boundary.shape[0]), :] - pref))).flatten())
        avg = (np.mean(boundary[i: i + step], axis=0))
        # 添加到平滑后的边界点
        boundary_smooth.append(avg)

    # 3、 对平滑后的边界点进行分析, 以确定峰谷点
    boundary_smooth = np.array(boundary_smooth).reshape(-1, 2)
    valley, _ = find_peaks(boundary_smooth[:, 0], height=0)

    # 4、更正正确的谷点
    ind = []
    for va in valley:
        ind.append(np.argmin(dist[va]) + va * step)

    # 返回谷点
    return boundary[ind, :]


def hand_valley_extract(mask, L=0.2, limit=50):
    """
    提取手掌的两个关键谷点
    :param mask: 原图像轮廓
    :param L: 下端到轮廓中心占图像高度的比例
    :param limit: 边界约束
    :return: 谷点1, 谷点2
    """
    # 拷贝图像
    img_ = mask.copy()

    # 计算轮廓中心
    """
    不要使用轮廓计算！ 仍有可能取到反向轮廓 导致轮廓中心具有较大偏差
    """
    contours, hierarchy = cv2.findContours(img_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(max_contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    """
    注: 轮廓下方L处可能位于图像外部, 故需要判断, 以进行修正
    """
    # 若轮廓下方L处位于图像外部, 则将下边界置 轮廓最下方与轮廓中心L比例处
    if cx + int(L * img_.shape[0]) >= img_.shape[0]:
        # 找到轮廓最下方的点
        x = np.max(max_contour[:, :, 1])
        # 下边界
        bottom_edge = int(cx * (1 - L) + x * L)
    else:
        # 否则 下边界在图像内部
        bottom_edge = cx + int(L * img_.shape[0])
    # 将轮廓中心下侧L部分的区域置为0
    img_[bottom_edge:, :] = 0

    # 确定Pref点
    y_ = np.nan
    row = 1
    # 从bottom_edge开始向上搜索, 直到找到第一个白色像素那行, 中点即为Pref点
    while np.isnan(y_):
        if row == bottom_edge:
            # 若搜索到了轮廓上方, 则认为图像质量不够好, 跳过
            return None, None
        p_ls = np.argwhere(img_[bottom_edge - row, ...] == 255)
        if len(p_ls) == 0:
            row += 1
            continue
        y_ = int(np.mean(p_ls))
        break
    # 此时得到Pref点坐标 (bottom_edge, y_)

    """
    由于该版本中 没有RDF函数 故考虑进行手搓一个差不多效果的函数
    考虑到手掌图像是竖直 故可以直接得到最左侧以及最右侧的点作为两个指尖
    在参考点中间部分容许范围内进行搜索 找到最远距离点作为中指指尖
    而两两指尖内部到Pref点距离的最小值作为谷点
    确定该两组谷点 作为基准线
    """
    # 先进行腐蚀操作 侵蚀掉大部分的干扰点
    kernel = np.ones((3, 3), np.uint8)
    img_ = cv2.dilate(img_, kernel, iterations=3)

    # 由于轮廓仍然可能识别成手掌外部, 故考虑以手掌为计算对象
    # 找到手掌最左侧的点
    left_points = np.argwhere(img_[:, limit: y_] == 255)
    left_most = left_points[left_points[:, 1].argmin()]
    # 补偿limit
    left_most[1] += limit
    # 找到手掌最右侧的点
    right_points = np.argwhere(img_[:, y_: img_.shape[1] - limit] == 255)
    right_most = right_points[right_points[:, 1].argmax()]
    # 补偿y_
    right_most[1] += y_
    # 找到手掌中指指尖
    middle_points = np.argwhere(img_[:, y_ - limit: y_ + limit] == 255)
    middle_most = middle_points[middle_points[:, 0].argmin()]
    # 补偿y_ - limit
    middle_most[1] += y_ - limit

    # 计算全部的谷点
    valleys = rdf_min(img_, left_most, right_most, (bottom_edge, y_))
    # 如果谷点数目小于2, 则认为图像质量不够好, 跳过
    if len(valleys) < 2:
        # print("valley can't be found, 图像质量不够好, 跳过")
        # # # 转换到bgr通道 以便画图
        # img__ = cv2.cvtColor(img_, cv2.COLOR_GRAY2BGR)
        # # 画出峰谷点
        # cv2.circle(img__, tuple([left_most[1], left_most[0]]), 5, (0, 0, 255), -1)
        # cv2.circle(img__, tuple([right_most[1], right_most[0]]), 5, (0, 0, 255), -1)
        # cv2.circle(img__, tuple([middle_most[1], middle_most[0]]), 5, (0, 0, 255), -1)
        # cv2.circle(img__, tuple([y_, bottom_edge]), 5, (0, 0, 255), -1)
        #
        # cv2.imshow('img', img__)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return None, None

    # 确定我们所需要的谷点
    # 由于谷点可能会有多个, 故需要进行筛选
    """
    根据规则:
    1、左侧第一个谷点即为我们所需要的谷点1
    2、依据谷点1, 在谷点1上侧临近范围内搜索谷点2, 且谷点2的y坐标应该在谷点1的y坐标上方附近
    3、在中指指尖两侧搜索谷点3
    """
    # 搜索谷点1
    valley_1 = valleys[0]
    # 搜索谷点2
    valley_2 = valley_1
    for valley in valleys[1:]:
        if valley[0] + img_.shape[0] * L > valley_1[0] > valley[0]:
            # if abs(valley[1] - middle_most[1]) < img_.shape[1] * L * 0.015:
            valley_2 = valley
            # 否则, 没有找到合适的谷点2, 则认为图像质量不够好, 跳过
    if valley_2[0] == valley_1[0]:
        return None, None

    # # 画出峰谷点
    # # # 转换到bgr通道 以便画图
    # img_ = cv2.cvtColor(img_, cv2.COLOR_GRAY2BGR)
    # # 绘制连接线
    # cv2.line(img_, tuple([valley_1[1], valley_1[0]]), tuple([valley_2[1], valley_2[0]]), (0, 0, 255), 2)

    # 返回两个谷点
    return valley_1, valley_2


def palm_roi_extract(img, desp="图像"):
    """
    :param img: 手掌图像
    :param mask: 手掌mask图像
    :return: 旋转之后的图像, 抽取的手掌ROI, mask
    """
    img_, mask = evaluate_pic_quality(img, desp)
    # 显示结果
    if img_ is None:
        print(f"{desp}质量不够好, 跳过")
        return None, None, None

    # 考察mask轮廓中心, 如果轮廓中心为黑色, 则变换颜色通道
    contours, hierarchy = cv2.findContours(img_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(max_contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    if mask[cy, cx] == 0:
        mask = 255 - mask

    valley_1, valley_2 = hand_valley_extract(mask)

    if valley_1 is None:
        # print(f"valley can't be found, {desp}质量不够好, 跳过")
        return None, None, None

    # 计算倾斜角
    angle = np.arctan((valley_2[0] - valley_1[0]) / (valley_2[1] - valley_1[1])) * 180 / np.pi

    # 旋转图像
    M = cv2.getRotationMatrix2D((int(valley_1[1]), int(valley_1[0])), angle, 1)
    rotated_img = cv2.warpAffine(img_, M, (img_.shape[1], img_.shape[0]))
    # 截取图像
    # 计算截取区域
    dist = int(np.sqrt(sum(np.square(valley_1 - valley_2))))
    # 起始像素为谷点1, 边长为dist
    valley_1[0] += 20
    # 故可以截取区域为该正方形
    res = rotated_img[valley_1[0]: valley_1[0] + dist, valley_1[1]: valley_1[1] + dist]

    return rotated_img, res, mask
