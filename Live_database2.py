import os
import cv2
from matplotlib import pyplot as plt
import numpy as np



class live_database2():

    basepath = './Live_database2'
    ff_path = os.path.join(basepath,'fastfading')
    gblur_path = os.path.join(basepath,'gblur')
    jp2k_path = os.path.join(basepath,'jp2k')
    jpeg_path = os.path.join(basepath,'jpeg')
    wn_path = os.path.join(basepath,'wn')

    ref_path = os.path.join(basepath,'refimgs')


    def get_degrade_ref_pic(self,folder):
        infopath = os.path.join(folder,'info.txt')
        dmospath = os.path.join(folder,'dmos.txt')
        dmos = []
        datalist = []
        fr = open(dmospath,'r')
        for line in fr:
            dmos = line.split(',')
            break


        fr = open(infopath,'r')
        for i,line in enumerate(fr):
            refimg_path = os.path.join(self.ref_path,line.split(' ')[0])
            deimg_path = os.path.join(folder,line.split(' ')[1])
            datalist.append([refimg_path,deimg_path,dmos[i]])
            #yield imgref,imgde,float(dmos[i])
        return datalist


if __name__ == '__main__':
    a = live_database2()
    for imgref,imgde,score in a.get_degrade_ref_pic(a.jp2k_path):
        plt.figure("DMOS:" + str(score))
        plt.subplot(1, 2, 1)
        plt.imshow(imgref, cmap='gray')
        plt.title("Ref. Image")
        plt.subplot(1, 2, 2)
        plt.imshow(imgde, cmap='gray')
        plt.title("Degraded Image")
        print 'score:', score
        plt.show()
    exit(0)



