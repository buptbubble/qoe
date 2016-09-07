# -*- coding: utf-8 -*-

from Tkinter import *
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import cv2
from PIL import Image
from PIL import ImageTk

class mainWindow():
    def __init__(self):
        self.root = Tk()

    def callback(self,event):
        print("Click position:", event.x, event.y)

    def quit(self):
        self.root.quit()
        self.root.destroy()

    def runfunc(self):
        pic = cv2.imread('pics/Lenna.png',cv2.CV_LOAD_IMAGE_GRAYSCALE)
        image = Image.fromarray(pic)
        image = ImageTk.PhotoImage(image)

        label1 = Label(image = image)
        label1.bind("<Button-1>", self.callback)
        label1.pack()


        mainloop()





if __name__ == '__main__':
    mainWin = mainWindow()
    mainWin.runfunc()






#
# root = Tk()
# def callback(event):
#     print("Click position:", event.x, event.y)
# frame = Frame(root, width=1000, height=600)
# frame.bind("<Button-1>", callback)
# frame.pack()
# label1 = Label(text = "test test test")
# label1.pack()
#
# mainloop()