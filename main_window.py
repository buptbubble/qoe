# -*- coding: utf-8 -*-

from Tkinter import *
import cv2
from PIL import Image
from PIL import ImageTk
from LSK_method import LSK_method
import numpy as np


def resizeByScale(img,scale):
    shape_ori = img.shape
    shape_new = [0, 0]
    shape_new[0] = shape_ori[0] * scale
    shape_new[1] = shape_ori[1] * scale
    return cv2.resize(img, tuple(shape_new), interpolation=cv2.INTER_NEAREST)

class mainWindow():
    def __init__(self):
        self.root = Tk()

        self.root.bind('<FocusIn>',self.focus_tk_child)
        self.tk_child = None
        self.picpath = 'pics/Lenna.png'

        self.runfunc()
    def mat2tkimg(self,matimg):
        return ImageTk.PhotoImage(Image.fromarray(matimg))


    def focus_tk_child(self,event):
        if self.tk_child != None:
            self.tk_child.focus()


    def callback(self,event):
        print '----------------------------------'
        print "Click position:", event.x-1, event.y-1

        clickp = np.array([event.y-1,event.x-1])
        if self.tk_child == None:
            self.tk_child = Toplevel(self.root,borderwidth=5)
            self.tk_child.attributes("-toolwindow",True)
            self.label0 = Label(master=self.tk_child,text= 'Weight Image')
            self.label0.grid(row=0)
            self.subtk_label1 = Label(master=self.tk_child)
            self.subtk_label1.grid(row=1)

            self.label2 = Label(master=self.tk_child,text= 'Gauss K Image').grid(row=2)
            self.label3 = Label(master=self.tk_child)
            self.label3.grid(row=3)

            self.label4 = Label(master=self.tk_child,text= 'ROI').grid(row=4)
            self.label5 = Label(master=self.tk_child)
            self.label5.grid(row=5)

            self.label6 = Label(master=self.tk_child,text='DeriX').grid(row=6)
            self.label7 = Label(master=self.tk_child)
            self.label7.grid(row=7)

            self.label8 = Label(master=self.tk_child, text='DeriY').grid(row=8)
            self.label9 = Label(master=self.tk_child)
            self.label9.grid(row=9)


        w_img = self.lsk.get_W_img(clickp,7)
        np.set_printoptions(precision=2,suppress=True)

        k_img = self.lsk.get_K_img(clickp,11)
        pic_roi = self.lsk.getROI(clickp,5)
        pic_derix = self.lsk.getDeriX(clickp,5)
        pic_deriy = self.lsk.getDeriY(clickp,5)



        w_img = cv2.normalize(w_img,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)
        k_img = cv2.normalize(k_img,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)

        pic_derix = cv2.normalize(pic_derix,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)
        pic_deriy = cv2.normalize(pic_deriy,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)

        w_img_scale = resizeByScale(w_img,10)
        k_img_scale = resizeByScale(k_img,10)
        pic_roi_scale = resizeByScale(pic_roi,10)
        pic_derix_scale = resizeByScale(pic_derix,10)
        pic_deriy_scale = resizeByScale(pic_deriy,10)






        self.weight_img = self.mat2tkimg(w_img_scale)
        self.k_img = self.mat2tkimg(k_img_scale)
        self.roi_img = self.mat2tkimg(pic_roi_scale)
        self.derix_img = self.mat2tkimg(pic_derix_scale)
        self.deriy_img = self.mat2tkimg(pic_deriy_scale)

        self.subtk_label1.configure(image=self.weight_img)
        self.label3.configure(image=self.k_img)
        self.label5.configure(image=self.roi_img)
        self.label7.configure(image = self.derix_img)
        self.label9.configure(image = self.deriy_img)


        self.lsk.printInfo(clickp)


        self.tk_child.focus()

    def quit(self):
        self.root.quit()
        self.root.destroy()

    def runfunc(self):
        self.pic = cv2.imread(self.picpath,cv2.CV_LOAD_IMAGE_GRAYSCALE)
        self.lsk = LSK_method(self.picpath)

        image = Image.fromarray(self.pic)
        image = ImageTk.PhotoImage(image)
        label1 = Label(self.root,image = image)
        label1.bind("<Button-1>", self.callback)


        label1.pack()

        mainloop()

if __name__ == '__main__':
    mainWin = mainWindow()

