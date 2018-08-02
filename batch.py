#coding=utf-8
# 直接代码调用，核心算法看这里即可
from calculate import *
# from calculate3 import *
from skimage import io
import imutils
import cv2
import os
import glob



def predeal(url,w=1080,is_gray=False):
    # cap=cv2.VideoCapture(url)
    # et,orig=cap.read()
    orig=io.imread(url)
    if(orig.shape[1]>w):
        orig=imutils.resize(orig, w)#缩小原图
    orig=cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    if is_gray:
        orig=cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    return orig

# def resize_w(orig,w=1080):
#     re=imutils.resize(orig, w)#缩小原图
#     return re
def reset(img_proc):
    img_proc.border_mask = np.zeros((img_proc.img.shape[0],img_proc.img.shape[1]),np.uint8)
    img_proc.border_mask[:,:] = 255
    img_proc.border_cnt = []

def show(img):
    cv2.imshow('dd',img)
    cv2.waitKey(0)

if __name__ == "__main__":  
    # WSI_MASK_PATH = 'eye-picture/input'#存放图片的文件夹路径  
    # INPUT_PATH = 'eye-picture/input/'#存放图片的文件夹路径  
    INPUT_PATH = 'input/'#存放图片的文件夹路径
    OUTPUT_PATH = 'output/'#存放图片的文件夹路径
    # OUTPUT_PATH = 'eye-picture/output/'#存放图片的文件夹路径  
    # OUTPUT_PATH = 'eye-picture/output/'#存放图片的文件夹路径  
    wsi_mask_paths = glob.glob(os.path.join(INPUT_PATH, '*.*'))
    # wsi_mask_paths = glob.glob(os.path.join(INPUT_PATH, '8.jpg'))
    wsi_mask_paths.sort()  
    imgls=[]
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    for img_path in wsi_mask_paths:
        print(img_path)
        # img_path="eye-picture/input_new/3.jpg"
        path,read_name=os.path.split(img_path)
        img_proc = ImgProcess()
        img_proc.ori_img=predeal(img_path)
        img_proc.img=predeal(img_path,320)

        reset(img_proc)
        imgs3=img_proc.calculate()

        name=OUTPUT_PATH+"done_"+read_name
        a=("%.2f%%" % (img_proc.dirt_area*100/img_proc.all_area))
        img_proc.final_img=cv2.putText(img_proc.final_img,a,(50,150),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),8)
        a=np.hstack([img_proc.ori_img,img_proc.final_img])
        # show(a)

        cv2.imwrite(name,a)

        # print(name)
        # show(np.vstack([imgs1,imgs2,imgs3]))
    # show(img_proc.final_img)
    # show(np.vstack([imgs1,imgs3]))
    # show(np.hstack(imgls))
    # img_proc.mask_img = cv2.resize(img_proc.final_img,(width, height))
    # img_proc.mask_img = cv2.cvtColor(img_proc.mask_img, cv2.COLOR_BGR2RGB)     
