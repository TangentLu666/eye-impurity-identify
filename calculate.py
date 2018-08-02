# coding=utf-8
import numpy as np
import cv2
import traceback
import random
import skimage.filters.rank as sfr
from skimage.morphology import disk


# from until import *

class ImgProcess():
    def __init__(self, parent=None):
        self.ori_img = None
        self.img = None
        self.mask_img = None
        self.mask = None
        self.final_img = None
        self.final_mask = None
        self.circle = None

        self.border_img = None
        self.border_mask = None
        self.border_cnt = []
        self.if_set_border = False

        self.dirt_area = 0
        self.all_area = 1

        self.orig_gray = None

    def show(self,img):
        cv2.namedWindow("res",0);
        cv2.resizeWindow("res", 1800, 960);
        cv2.moveWindow("res", 100, 50);
        cv2.imshow("res",img)
        cv2.waitKey(0)

    def distance(self, p1, p2):
        dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        return dist

    def filter_border(self, ori_img, img):
        if self.if_set_border:
            x = int(np.mean(self.border_cnt, axis=0)[0])
            y = int(np.mean(self.border_cnt, axis=0)[1])
            # cv2.imshow('before', self.border_mask)
            self.border_mask = cv2.fillConvexPoly(self.border_mask, self.border_cnt, 128)
            color = self.border_mask[y][x]
            # self.border_mask[y][x] = 255
            mask = cv2.resize(self.border_mask, (ori_img.shape[1], ori_img.shape[0]),
                              cv2.INTER_NEAREST)
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if color != mask[i][j]:
                        self.all_area += 1
                        if 255 == self.final_mask[i][j]:
                            for c in range(3):
                                img[i][j][c] = ori_img[i][j][c]
                            self.final_mask[i][j] = 0

            # cv2.imshow('after', self.border_mask)
            # cv2.waitKey(0)
        return img

    def auto_canny(self, image, sigma=0.7):
        v = np.median(image)
        upper = int(min(255, sigma * v))
        lower = int(upper / 4)
        edged = cv2.Canny(image, lower, upper)
        return edged

    def filter_light(self, ori_img, img):
        light_thresh = 180
        thresh = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        for i in range(thresh.shape[0]):
            for j in range(thresh.shape[1]):
                if img[i][j][0] > light_thresh and img[i][j][1] > light_thresh and img[i][j][
                    2] > light_thresh:
                    thresh[i][j] = 255
        # cv2.imshow('before', thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        thresh = cv2.dilate(thresh, kernel, 5)
        # cv2.imshow('after', thresh)

        for i in range(thresh.shape[0]):
            for j in range(thresh.shape[1]):
                if thresh[i][j] == 255:
                    if self.final_mask[i][j] == 255:
                        self.final_mask[i][j] = 0
                        for c in range(3):
                            img[i][j][c] = ori_img[i][j][c]

        return img

    def func2(self, i, j):
        return self.distance([i, j], [self.circle[1], self.circle[0]]) < self.circle[2]

    def merge(self, ori_img, mask_img, fil_light=True):
        img = ori_img.copy()
        img_s = cv2.split(img)
        if not self.if_set_border:
            print("merge base circle")
            # hough圆范围矩阵
            border = np.fromfunction(self.func2, mask_img.shape)
        else:
            print("merge base border")
            self.border_mask = cv2.fillConvexPoly(self.border_mask, self.border_cnt, 0)
            cv2.imwrite("out.jpg", self.border_mask)
            mask = cv2.resize(self.border_mask, (ori_img.shape[1], ori_img.shape[0]),
                              cv2.INTER_NEAREST)
            # 手选范围矩阵
            border = np.where(mask == 0, True, False)
        # 计算总面积
        self.all_area = np.sum(border == True)
        # 叠加检测结果矩阵
        border = np.where(mask_img > 0, border, False)

        # 叠加局部细纹合并矩阵（局部细纹大于30%认定整个区域为杂质区域）
        bad = sfr.mean(border, disk(15))
        border = np.where(bad > 255*0.30, True, border) # 原0.25
        border = np.where(bad < 255*0.08, False, border) # 原0.08

        if fil_light:  # 叠加过滤光斑矩阵
            light = sfr.maximum(self.orig_gray, disk(15))
            # light_mean = sfr.mean(self.orig_gray, disk(15))
            # light=np.where(light_mean>255*0.9,light,self.orig_gray)
            border = np.where(light < 230, border, False)

        mask=np.where(border, mask_img, 0)
        img_s[0] = np.where(border, 255, img_s[0])
        img_s[1] = np.where(border, 0, img_s[1])
        img_s[2] = np.where(border, 0, img_s[2])
        img = cv2.merge(img_s)
        # 计算杂质面积
        self.dirt_area = np.sum(border == True)
        return img,mask

    def add_border(self, x, y, size=1):
        self.border_mask[max(0, y - size): min(y + size, self.border_mask.shape[0] - size), \
        max(0, x - size): min(x + size, self.border_mask.shape[1] - size)] = 0
        self.border_img[max(0, y - size): min(y + size, self.border_mask.shape[0] - size), \
        max(0, x - size): min(x + size, self.border_mask.shape[1] - size)] = (0, 255, 0)
        self.border_cnt.append(np.array([x, y]))

    def add_point(self, x, y, size=1):
        if 255 == self.mask[y, x]:
            return False
        self.mask[max(0, y - size): min(y + size, self.mask.shape[0] - size), \
        max(0, x - size): min(x + size, self.mask.shape[1] - size)] = 255
        self.mask_img[max(0, y - size): min(y + size, self.mask.shape[0] - size), \
        max(0, x - size): min(x + size, self.mask.shape[1] - size)] = (0, 0, 255)
        return True

    def erase_point(self, x, y, size=4):
        erase_num = np.sum(self.mask[max(0, y - size): min(y + size, self.mask.shape[0] - size), \
                           max(0, x - size): min(x + size, self.mask.shape[1] - size)] > 0)
        if 0 == erase_num:
            return False
        self.mask[max(0, y - size): min(y + size, self.mask.shape[0] - size), \
        max(0, x - size): min(x + size, self.mask.shape[1] - size)] = 0

        self.mask_img[max(0, y - size): min(y + size, self.mask.shape[0] - size), \
        max(0, x - size): min(x + size, self.mask.shape[1] - size)] = \
            self.img[max(0, y - size): min(y + size, self.mask.shape[0] - size), \
            max(0, x - size): min(x + size, self.mask.shape[1] - size)]

        return erase_num

    def save_process(self):
        mask = cv2.resize(self.mask, (self.ori_img.shape[1], self.ori_img.shape[0]),
                          cv2.INTER_NEAREST)
        self.final_img[:,:,0]=np.where(mask>0,mask,self.ori_img[:,:,0])
        self.final_img[:,:,1]=np.where(mask>0,0,self.ori_img[:,:,1])
        self.final_img[:,:,2]=np.where(mask>0,0,self.ori_img[:,:,2])
        # for i in range(mask.shape[0]):
        #     for j in range(mask.shape[1]):
        #         if 255 == mask[i][j]:
        #             self.final_img[i][j] = (255, 0, 0)
        #         else:
        #             for c in range(3):
        #                 self.final_img[i][j][c] = self.ori_img[i][j][c]

    def calculate(self):
        self.dirt_area = 0
        img = self.ori_img.copy()
        self.final_img = img
        self.final_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        self.mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        self.border_cnt = np.array(self.border_cnt)

        ori_img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            print('calculating...')
            thresh = np.zeros(gray.shape, np.uint8)
            if not self.if_set_border:
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 600,
                                           param1=50, param2=30, minRadius=250, maxRadius=0)
                circles = np.uint16(np.around(circles))
                circle = circles[0][0]
                gray_roi = gray[
                           max(0, circle[1] - circle[2]): min(circle[1] + circle[2], gray.shape[0]),
                           max(0, circle[0] - circle[2]): min(circle[0] + circle[2], gray.shape[1])]
                thresh_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                                   cv2.THRESH_BINARY, 11, 8)
                thresh[max(0, circle[1] - circle[2]): min(circle[1] + circle[2], gray.shape[0]),
                max(0, circle[0] - circle[2]): min(circle[0] + circle[2],
                                                   gray.shape[1])] = thresh_roi

            else:
                scale = 1.0 * self.border_mask.shape[0] / img.shape[0]
                gray_roi = gray[int(np.min(self.border_cnt[:, 0] / scale)): \
                                int(np.max(self.border_cnt[:, 0] / scale)), \
                           int(np.min(self.border_cnt[:, 1] / scale)): \
                           int(np.max(self.border_cnt[:, 1] / scale))]
                thresh_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                                   cv2.THRESH_BINARY, 11, 8)
                thresh[int(np.min(self.border_cnt[:, 0] / scale)): \
                       int(np.max(self.border_cnt[:, 0] / scale)), \
                int(np.min(self.border_cnt[:, 1] / scale)): \
                int(np.max(self.border_cnt[:, 1] / scale))] = thresh_roi
            _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                                      cv2.CHAIN_APPROX_SIMPLE)
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area < 200:
                    img = cv2.fillConvexPoly(img, cnt, (255, 0, 0))
                    self.final_mask = cv2.fillConvexPoly(self.final_mask, cnt, (255))

            if not self.if_set_border:
                for i in range(thresh.shape[0]):
                    for j in range(thresh.shape[1]):
                        if (255 == self.final_mask[i][j]) and (
                                self.distance([i, j], [circle[1], circle[0]]) > circle[2]):
                            for c in range(3):
                                img[i][j][c] = ori_img[i][j][c]
                            self.final_mask[i][j] = 0
                self.all_area = int(int(circle[2]) * circle[2] * 3.14)

            img = self.filter_border(ori_img, img)
            self.final_img = self.filter_light(ori_img, img)

            for i in range(self.final_mask.shape[0]):
                for j in range(self.final_mask.shape[1]):
                    if self.final_mask[i][j] == 255:
                        self.dirt_area += 1
            self.mask = cv2.resize(self.final_mask,
                                   (self.border_mask.shape[1], self.border_mask.shape[0]),
                                   cv2.INTER_NEAREST)

            # cv2.imshow('ori_img', self.ori_img)
            # cv2.waitKey(0)


        except:
            traceback.print_exc()

        print('done')

    # --------------2-----------2----------2-----------2----------2-----------2----------2-----------2----------2-----------2----------2
    def calculate2(self, canny=0.1, mid=5):
        self.dirt_area = 0
        img = self.ori_img.copy()
        self.final_img = img
        self.final_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        self.mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        self.border_cnt = np.array(self.border_cnt)

        ori_img = img.copy()
        img = cv2.medianBlur(img, mid)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.orig_gray = gray
        try:
            print('calculating2...')
            thresh = np.zeros(gray.shape, np.uint8)
            if not self.if_set_border:
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 600,
                                           param1=80, param2=20, minRadius=250, maxRadius=0)
                circles = np.uint16(np.around(circles))
                circle = circles[0][0]
                self.circle = circle
                gray_roi = gray[
                           max(0, circle[1] - circle[2]): min(circle[1] + circle[2], gray.shape[0]),
                           max(0, circle[0] - circle[2]): min(circle[0] + circle[2], gray.shape[1])]
                thresh_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                                   cv2.THRESH_BINARY_INV, 13, 2)
                # thresh_roi=self.auto_canny(gray_roi,canny)
                thresh[max(0, circle[1] - circle[2]): min(circle[1] + circle[2], gray.shape[0]),
                max(0, circle[0] - circle[2]): min(circle[0] + circle[2],
                                                   gray.shape[1])] = thresh_roi

            else:
                scale = 1.0 * self.border_mask.shape[0] / img.shape[0]
                gray_roi = gray[int(np.min(self.border_cnt[:, 1] / scale)): \
                                int(np.max(self.border_cnt[:, 1] / scale)), \
                           int(np.min(self.border_cnt[:, 0] / scale)): \
                           int(np.max(self.border_cnt[:, 0] / scale))]
                thresh_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                                   cv2.THRESH_BINARY_INV, 13, 2)
                thresh[int(np.min(self.border_cnt[:, 1] / scale)): \
                       int(np.max(self.border_cnt[:, 1] / scale)), \
                int(np.min(self.border_cnt[:, 0] / scale)): \
                int(np.max(self.border_cnt[:, 0] / scale))] = thresh_roi

            # --------------2-----------2----------2-----------2----------2-----------2----------2-----------2----------2-----------2----------2
            self.final_img,self.final_mask = self.merge(img, thresh)
            # self.final_img = self.filter_border(ori_img, img)
            # img = self.filter_border(ori_img, img)
            # self.final_img = self.filter_light(ori_img, img)

            # for i in range(self.final_mask.shape[0]):
            #     for j in range(self.final_mask.shape[1]):
            #         if self.final_mask[i][j] == 255:
            #             self.dirt_area+=1

            self.mask = cv2.resize(self.final_mask,(self.border_mask.shape[1], self.border_mask.shape[0]), cv2.INTER_NEAREST)

        except:
            traceback.print_exc()

        print('done')

    # --------------3-----------3----------3-----------3----------3-----------3----------3-----------3----------3-----------3----------3
    def calculate3(self):
        self.dirt_area = 0
        img = self.ori_img.copy()
        self.final_img = img
        self.final_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        self.mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        self.border_cnt = np.array(self.border_cnt)

        ori_img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        try:
            print('calculate_bad2...')
            thresh = np.zeros(gray.shape, np.uint8)
            if not self.if_set_border:
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 600,
                                           param1=100, param2=20, minRadius=250, maxRadius=0)
                circles = np.uint16(np.around(circles))
                circle = circles[0][0]
                self.circle = circle

                gray_roi = gray[
                           max(0, circle[1] - circle[2]): min(circle[1] + circle[2], gray.shape[0]),
                           max(0, circle[0] - circle[2]): min(circle[0] + circle[2], gray.shape[1])]
                thresh_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                                   cv2.THRESH_BINARY, 233, 3)
                thresh[max(0, circle[1] - circle[2]): min(circle[1] + circle[2], gray.shape[0]),
                max(0, circle[0] - circle[2]): min(circle[0] + circle[2],
                                                   gray.shape[1])] = thresh_roi

            else:
                scale = 1.0 * self.border_mask.shape[0] / img.shape[0]
                gray_roi = gray[int(np.min(self.border_cnt[:, 1] / scale)): \
                                int(np.max(self.border_cnt[:, 1] / scale)), \
                           int(np.min(self.border_cnt[:, 0] / scale)): \
                           int(np.max(self.border_cnt[:, 0] / scale))]
                thresh_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                                   cv2.THRESH_BINARY, 233, 3)

                thresh[int(np.min(self.border_cnt[:, 1] / scale)): \
                       int(np.max(self.border_cnt[:, 1] / scale)), \
                int(np.min(self.border_cnt[:, 0] / scale)): \
                int(np.max(self.border_cnt[:, 0] / scale))] = thresh_roi


                # scale = 1.0 * self.border_mask.shape[0] / img.shape[0]
                # gray_roi = gray[int(np.min(self.border_cnt[:, 1] / scale)): \
                #                 int(np.max(self.border_cnt[:, 1] / scale)), \
                #            int(np.min(self.border_cnt[:, 0] / scale)): \
                #            int(np.max(self.border_cnt[:, 0] / scale))]
                #
                #
                # thresh[int(np.min(self.border_cnt[:, 1] / scale)): \
                #        int(np.max(self.border_cnt[:, 1] / scale)), \
                # int(np.min(self.border_cnt[:, 0] / scale)): \
                # int(np.max(self.border_cnt[:, 0] / scale))] = thresh_roi


            print('set_borde...')
            self.final_img,self.final_mask = self.merge(ori_img, thresh, False)
            # for i in range(self.final_mask.shape[0]):
            #     for j in range(self.final_mask.shape[1]):
            #         if self.final_mask[i][j] == 255:
            #             self.dirt_area+=1
            self.mask = cv2.resize(self.final_mask,(self.border_mask.shape[1], self.border_mask.shape[0]), cv2.INTER_NEAREST)
            # print(self.dirt_area)
            # print(np.sum())
            # cv2.imshow('ori_img', self.ori_img)
            # cv2.waitKey(0)

        except:
            traceback.print_exc()

        print('done')
