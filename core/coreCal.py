# coding=utf-8
# 核心运算代码
import traceback
import numpy as np
import cv2

def calculate(ori_img, canny=0.1, mid=5):
    img = ori_img.copy() # 拷贝一份原图
    img = cv2.medianBlur(img, mid) # 先进行一次中值滤波，消除部分噪点
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 把图片转成灰度
    try:
        print('calculating...')
        thresh = np.zeros(gray.shape, np.uint8) # 按照原图大小生成一张全黑的单通道图片(数组)
        # 使用opencv的霍夫变换先找出图像所有可能是眼球的圆
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 600,
                                   param1=80, param2=20, minRadius=250, maxRadius=0)
        # 整理数组格式
        circles = np.uint16(np.around(circles))
        # 选取第一个圆为眼球(笔者测试了多次，发现第一个基本都是圆心最靠近中心，且直径最大)

        global circle
        circle = circles[0][0]

        # 以圆心为中心，分别在x、y方向加减半径，得到圆内切正方形区域(考虑图片边界，故需要做max、min运算)
        gray_roi = gray[
                   max(0, circle[1] - circle[2]): min(circle[1] + circle[2], gray.shape[0]),
                   max(0, circle[0] - circle[2]): min(circle[0] + circle[2], gray.shape[1])]
        # 对正方形区域进行自适应二值化运算，得到颜色变化较大的区域标识(不平整的眼球表面,即眼球杂质) img_01.png
        thresh_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                           cv2.THRESH_BINARY_INV, 13, 2)
        # 把眼球杂志结果赋值到原图大小的纯黑图上
        thresh[max(0, circle[1] - circle[2]): min(circle[1] + circle[2], gray.shape[0]),
        max(0, circle[0] - circle[2]): min(circle[0] + circle[2],
                                           gray.shape[1])] = thresh_roi
        # 进行合并
        return merge(img, thresh)
    except:
        traceback.print_exc()

    print('done')

# 图片合并，过滤灯光
def merge(img, mask_img):
    # 把原图分隔成三通道图
    img_s = cv2.split(img)
    # # 圆边界计算，得到一个坐标在眼球圆内的值为True，圆外为False的原图同大小数组 img_04.png
    border = np.fromfunction(func2, mask_img.shape)
    # 叠加检测结果矩阵，得到一个坐标眼球圆内，且有杂质标记的值为True的数组，也就是标记哪些像素是应该标记出来的
    border = np.where(mask_img > 0, border, False) # 把border改成True即可达到 img_05.png
    # 计算每个通道的值(255,0,0)为蓝色，即杂质部分标记为蓝色，其他部分使用原图
    img_s[0] = np.where(border, 255, img_s[0])
    img_s[1] = np.where(border, 0, img_s[1])
    img_s[2] = np.where(border, 0, img_s[2])
    # 合并三个通道得到最终图
    img = cv2.merge(img_s)
    return img


def func2(i, j):
    return distance([i, j], [circle[1], circle[0]]) < circle[2]

def distance(p1, p2):
    dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    return dist
