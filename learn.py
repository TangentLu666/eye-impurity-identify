# coding=utf-8

import traceback
import numpy as np
import cv2


# 展示结果
def show_res(file_name, imgs):
    res = np.hstack(imgs)
    # show_img(res) # 也可使用该方法直接弹窗显示
    cv2.imwrite(file_name + '.png', res)


# 弹窗显示图片
def show_img(img):
    cv2.imshow('show', img)
    cv2.waitKey(0)


# 4.1 中值滤波
def medianBlur(img, d=15):
    # 中值滤波，15是取范围，必须为奇数
    mid = cv2.medianBlur(img, d)
    show_res('medianBlur', [img, mid])
    return mid


# 4.2 霍夫圆
def hough_circle(img):
    # 转换成灰度图片
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 使用opencv的霍夫变换先找出图像所有可能是眼球的圆
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 600,
                               param1=80, param2=20, minRadius=250, maxRadius=0)
    # 拷贝一份原图
    res = img.copy()
    for c in circles:
        for x, y, r in c:
            cv2.circle(res, (x, y), r, (0, 0, 255), 2)
    show_res('hough_circle', [img, res])
    return circles[0][0]  # 取第一个圆效果最好


# 4.3 自适应二值化
def thresh(img):
    # 转换成灰度图片
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 使用固定阈值对gray图像进行二值化处理，125为阈值，高于该阈值则把像素点置为255，ret为返回的阈值，thresh1返回的结果
    ret, thresh_const = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
    # 支持两种自适应方法，即cv2.ADAPTIVE_THRESH_MEAN_C（平均）和cv2.ADAPTIVE_THRESH_GAUSSIAN_C（高斯）
    # 如果使用平均的方法，则所有像素周围的权值相同；如果使用高斯的方法，则（x,y）周围的像素的权值则根据其到中心点的距离通过高斯方程得到。
    # 在两种情况下，自适应阈值T(x, y)。通过计算每个像素周围bxb(b=13)大小像素块的加权均值并减去常量C=2得到。其中，b由blockSize给出，大小必须为奇数；
    thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                            cv2.THRESH_BINARY_INV, 13, 2)
    show_res('thresh', [gray, thresh_const, thresh_adaptive])
    return thresh_adaptive

# 4.4 结果识别
def calculate_res(img):
    # 中值滤波
    img = medianBlur(img, d=5)
    # 获得自适应二值化结果
    thresh_adaptive = thresh(img)
    # 获得霍夫圆,计算距离distance方法要用,我不知道怎么往里面传值，只能用全局变量了囧
    global circle
    circle = hough_circle(img)

    # 把在眼球内且为自适应二值化检测结果的像素标记为蓝色
    # 把原图按通道分割成一个数组
    img_s = cv2.split(img)
    # 圆边界计算，得到一个坐标在眼球圆内的值为True，圆外为False的原图同大小数组
    border = np.fromfunction(func, thresh_adaptive.shape)
    # 叠加检测结果矩阵，得到一个坐标眼球圆内，且有杂质标记的值为True的数组，也就是标记哪些像素是应该标记出来的
    border = np.where(thresh_adaptive > 0, border, False)
    # 计算每个通道的值(255,0,0)为蓝色，即杂质部分标记为蓝色，其他部分使用原图
    img_res = merge(border, img_s)

    # 把眼球内的像素就标记为蓝色【演示用】
    img_s1 = cv2.split(img)
    border1 = np.fromfunction(func, thresh_adaptive.shape)
    img_res1 = merge(border1, img_s1)

    # 把自适应二值化的检测结果像素标记为蓝色【演示用】
    img_s2 = cv2.split(img)
    border2 = np.where(thresh_adaptive > 0, True, False)
    img_res2 = merge(border2, img_s2)
    show_res('calculate_res', [img, img_res1, img_res2, img_res])

# 根据条件合并原图，把杂质标记成（255，0，0）颜色
def merge(border, img_s):
    img_s[0] = np.where(border, 255, img_s[0])
    img_s[1] = np.where(border, 0, img_s[1])
    img_s[2] = np.where(border, 0, img_s[2])
    # 合并三个通道得到最终图
    img_res = cv2.merge(img_s)
    return img_res

# 计算像素坐标是否在圆范围，返回布朗值
def func(i, j):
    return distance([i, j], [circle[1], circle[0]]) < circle[2]

def distance(p1, p2):
    dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    return dist


if __name__ == "__main__":
    # 读取图片
    orig_img = cv2.imread('1.png')
    show_img(orig_img)
    # # 4.1 中值滤波
    # medianBlur(orig_img)
    # # 4.2 霍夫圆
    # hough_circle(orig_img)
    # # 4.3 自适应二值化
    # thresh(orig_img)
    # 4.4 结果识别
    # calculate_res(orig_img)
