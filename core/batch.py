#coding=utf-8
# 批量处理
from skimage import io
import imutils
import cv2
import os
import glob
import numpy as np
from core.coreCal import calculate


def predeal(url,w=1080,is_gray=False):
    orig=io.imread(url)
    if(orig.shape[1]>w):
        orig=imutils.resize(orig, w)#缩小原图
    orig=cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    if is_gray:
        orig=cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    return orig

def show(img):
    cv2.imshow('dd',img)
    cv2.waitKey(0)

if __name__ == "__main__":  
    INPUT_PATH = 'core/input/'#存放图片的文件夹路径
    OUTPUT_PATH = 'core/output/'#存放图片的文件夹路径
    wsi_mask_paths = glob.glob(os.path.join(INPUT_PATH, '*.*'))
    wsi_mask_paths.sort()
    imgls=[]
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    for img_path in wsi_mask_paths:
        print(img_path)
        path,read_name=os.path.split(img_path)

        ori_img=predeal(img_path)
        res_img=calculate(ori_img)

        name=OUTPUT_PATH+"done_"+read_name

        # cv2.imwrite(name,res_img)
        cv2.imwrite(name,np.hstack([ori_img,res_img]))

