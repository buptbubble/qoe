# -*- coding: utf-8 -*-

from Tkinter import *
import cv2
from PIL import Image
from PIL import ImageTk

class mainWindow():
    def __init__(self):
        self.root = Tk()
        self.tk_child = None

    def mat2tkimg(self,matimg):
        return ImageTk.PhotoImage(Image.fromarray(matimg))

    def callback(self,event):
        print("Click position:", event.x, event.y)
        clickx = event.x
        clicky = event.y
        if self.tk_child == None:
            self.tk_child = Toplevel(self.root,borderwidth=5)
            self.tk_child.attributes("-toolwindow",True)
            self.subtk_label1 = Label(master=self.tk_child)
            self.subtk_label1.pack()
        mat_selected = self.pic[clicky:clicky+20,clickx:clickx+20]
        self.image_sel = self.mat2tkimg(mat_selected)
        self.subtk_label1.configure(image= self.image_sel)
        self.tk_child.focus()

    def quit(self):
        self.root.quit()
        self.root.destroy()

    def runfunc(self):
        self.pic = cv2.imread('pics/Lenna.png',cv2.CV_LOAD_IMAGE_GRAYSCALE)
        image = Image.fromarray(self.pic)
        image = ImageTk.PhotoImage(image)
        label1 = Label(self.root,image = image)
        label1.bind("<Button-1>", self.callback)
        label1.pack()

        mainloop()

if __name__ == '__main__':
    mainWin = mainWindow()
    mainWin.runfunc()
