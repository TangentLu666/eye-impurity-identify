#coding:utf-8
# 视图操作界面
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap,  QImage
from PyQt5.QtWidgets import *
import os
from sample.ImgProcess import *

dialog = None

class BtnLabel(QLabel):  
    def __init__(self,parent=None):  
        super(BtnLabel,self).__init__(parent)  
        self.if_mouse_press = False

    def mouseMoveEvent(self,e):  
        print ('mouse move:(%d,%d)\n'%(e.pos().x(),e.pos().y()))
        if self.if_mouse_press:
            dialog.move_point(e.pos().x(),e.pos().y())
    def mousePressEvent(self,e):  
        print ('mousePressEvent(%d,%d)\n'%(e.pos().x(),e.pos().y()))
        self.if_mouse_press = True
        dialog.move_point(e.pos().x(),e.pos().y())

    def mouseReleaseEvent(self,e):  
        print ('mouseReleaseEvent(%d,%d)\n'%(e.pos().x(),e.pos().y()))
        self.if_mouse_press = False


class MainDialog(QDialog):  
    def __init__(self,parent=None):  
        super(MainDialog,self).__init__(parent)  

        self.scale = 1
        self.width = 640
        self.height = 1
        self.bytesPerLine = self.width * 3
        self.img_proc = ImgProcess()
        self.if_calculated = False
        self.paint = None
        self.if_load_img = False
        self.path = '.'


        self.input_label = BtnLabel(self)  
        self.input_label.setGeometry(160, 40, 640, 480)  

        self.output_label = BtnLabel(self)  
        self.output_label.setGeometry(900, 40, 640, 480)  


        #set open file button
        self.open_btn = QtWidgets.QPushButton(self)  
        self.open_btn.setObjectName("open_btn")  
        self.open_btn.setGeometry(1, 0, 100, 40)

        self.open_btn.setText("open")  
        self.open_btn.clicked.connect(self.open_file) 

        #set calculate button
        self.calc_btn2 = QtWidgets.QPushButton(self)
        self.calc_btn2.setObjectName("calc_btn")
        self.calc_btn2.setGeometry(1, 90, 100, 40)
        self.calc_btn2.setText("calculate")
        self.calc_btn2.clicked.connect(self.calculate)

        #set save file button
        self.save_btn = QtWidgets.QPushButton(self)  
        self.save_btn.setObjectName("save_btn")  
        self.save_btn.setGeometry(1, 180, 100, 40)

        self.save_btn.setText("save")  
        self.save_btn.clicked.connect(self.save_file)  

        #set add point button
        self.add_point_btn = QtWidgets.QPushButton(self)  
        self.add_point_btn.setObjectName("add_point_btn")  
        self.add_point_btn.setGeometry(1, 240, 100, 40)

        self.add_point_btn.setText("add point")  
        self.add_point_btn.clicked.connect(self.add_point_on_click)  

        #set erase point button
        self.erase_point_btn = QtWidgets.QPushButton(self)  
        self.erase_point_btn.setObjectName("erase_point_btn")  
        self.erase_point_btn.setGeometry(1, 300, 100, 40)

        self.erase_point_btn.setText("erase point")  
        self.erase_point_btn.clicked.connect(self.erase_point_on_click)  

        #set add border button
        self.add_border_btn = QtWidgets.QPushButton(self)  
        self.add_border_btn.setObjectName("add_border_btn")  
        self.add_border_btn.setGeometry(1, 360, 100, 40)

        self.add_border_btn.setText("add border")  
        self.add_border_btn.clicked.connect(self.add_border_on_click)  

        #set result text
        self.text_label = QLabel(self)
        self.text_label.setAlignment(Qt.AlignCenter)  
        self.text_label.setGeometry(1, 420, 140, 100)



        self.text_label_b = QLabel(self)
        self.text_label_b.setAlignment(Qt.AlignCenter)
        self.text_label_b.setGeometry(1, 720, 140, 100)


    def add_point_on_click(self):
        self.paint = 'add_point'

    def erase_point_on_click(self):
        self.paint = 'erase_point'

    def add_border_on_click(self):
        self.reset()
        self.paint = 'add_border'



    def reset(self):
        self.paint = None
        self.circle = None
        self.img_proc.if_set_border = False
        self.img_proc.border_img = self.img_proc.img
        self.img_proc.border_mask = np.zeros((self.img_proc.img.shape[0],\
            self.img_proc.img.shape[1]),np.uint8)
        self.img_proc.border_mask[:,:] = 255
        self.img_proc.border_cnt = []

        img = self.img_proc.ori_img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                                           

        self.img_proc.img = cv2.resize(img,(self.width, self.height))

        QImg = QImage(self.img_proc.img.data, self.width, self.height, self.bytesPerLine,QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        self.input_label.setGeometry(160, 40, self.width, self.height)  
        self.input_label.setPixmap(pixmap)

    def update_text(self):
        self.text_label.setText("总面积: "+ str(self.img_proc.all_area) + '\n'\
            +'杂质面积: ' + str(int(self.img_proc.dirt_area)) + '\n'\
            + '百分比: ' + str(round(self.img_proc.dirt_area*100.0/self.img_proc.all_area,5)) + '%')  


    def open_file(self):  
        fileName1, filetype = QFileDialog.getOpenFileName(self,  
                                    "选取文件夹",  
                                    self.path,
                                    "Image Files(*.jpg *.png)")
        if len(fileName1) == 0:
            return
        self.path, read_name = os.path.split(fileName1)
        self.load_img(fileName1)
        self.output_label.setGeometry(self.width + 160, 40, self.width, self.img_proc.img.shape[0])  
        self.if_calculated = False
        self.reset()


    def save_file(self):  
        if self.if_calculated:
            file_name, status = QFileDialog.getSaveFileName(self,  
                                        "文件保存",  
                                        "./",  
                                        "Image Files(*.jpg *.png)")  
            # self.img_proc.save_process()
            try :
                # cv2.imwrite(file_name, self.img_proc.final_img)
                save_img=cv2.cvtColor(self.img_proc.mask_img, cv2.COLOR_BGR2RGB)
                a=("%.2f%%" % (self.img_proc.dirt_area*100/self.img_proc.all_area))
                save_img=cv2.putText(save_img,a,(20,60),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),4)
                cv2.imwrite(file_name, save_img)
            except :
                print ('save file error')

    def calculate(self):

        self.img_proc.calculate()
        self.img_proc.mask_img = cv2.resize(self.img_proc.final_img,(self.width, self.height))
        self.img_proc.mask_img = cv2.cvtColor(self.img_proc.mask_img, cv2.COLOR_BGR2RGB)                                           
        
        QImg = QImage(self.img_proc.mask_img.data, self.width, self.height, self.bytesPerLine,QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        self.output_label.setPixmap(pixmap)
        self.if_calculated = True
        self.reset()
        self.update_text()


    def move_point(self, x, y):
        if self.if_calculated and self.paint == 'add_point':
            self.add_point(x, y)
        if self.if_calculated and self.paint == 'erase_point':
            self.erase_point(x, y)
        if self.if_load_img and self.paint == 'add_border':
            self.add_border(x, y)

    def add_point(self, x, y):
        if self.img_proc.add_point(x, y):
            QImg = QImage(self.img_proc.mask_img.data, self.width, self.height, self.bytesPerLine,QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(QImg)
            self.output_label.setPixmap(pixmap)
            print(self.scale)
            self.img_proc.dirt_area += 1.0 / (self.scale * self.scale)
            self.update_text()
            


    def erase_point(self, x, y):
        erase_num = self.img_proc.erase_point(x, y)
        if erase_num:
            QImg = QImage(self.img_proc.mask_img.data, self.width, self.height, self.bytesPerLine,QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(QImg)
            self.output_label.setPixmap(pixmap)
            self.img_proc.dirt_area -= erase_num/ (self.scale * self.scale)
            self.update_text()
            

    def add_border(self, x, y):
        self.img_proc.add_border(x, y)
        QImg = QImage(self.img_proc.border_img.data, self.width, self.height, self.bytesPerLine,QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        self.input_label.setPixmap(pixmap)
        self.img_proc.if_set_border = True



    def load_img(self, img_path):
        img = cv2.imread(img_path)
        self.img_proc.ori_img = img
        height, width, bytesPerComponent = img.shape
        if width == 0 or height == 0:
            print ('load image faild: %s'%img_path)
            return 
        self.scale = self.width*1.0 / width
        height = int(height * self.scale)
        width = self.width
        img = cv2.resize(img,(width, height))
        self.img_proc.img = img
        self.height, self.width, bytesPerComponent = img.shape

        self.if_load_img = True


if __name__ == "__main__":  
    import sys  
    # app = QApplication(sys.argv)  
    # qb = MainLabel()  
    # qb.show()  
    # sys.exit(app.exec_())
    app=QApplication(sys.argv)  
    dialog=MainDialog()  
    dialog.show()  
    sys.exit(app.exec_())