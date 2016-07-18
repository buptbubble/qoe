import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import random
import re



class live_database2():

    basepath = 'Live_database2'
    ff_path = os.path.join(basepath,'fastfading')
    gblur_path = os.path.join(basepath,'gblur')
    jp2k_path = os.path.join(basepath,'jp2k')
    jpeg_path = os.path.join(basepath,'jpeg')
    wn_path = os.path.join(basepath,'wn')

    ref_path = os.path.join(basepath,'refimgs')


    def get_degrade_ref_pic(self,folder):

        degrationType = os.path.split(folder)[1]
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
            result = re.search("img(\d+)",deimg_path)
            picidx = int(result.group(1))
            datalist.append([refimg_path,deimg_path,float(dmos[picidx-1]),degrationType])
            #yield imgref,imgde,float(dmos[i])
        return datalist

    def get_picinfo_all_folder(self,sample_one_folder):
        jp2k_info = random.sample(self.get_degrade_ref_pic(self.jp2k_path),sample_one_folder)
        jpeg_info = random.sample(self.get_degrade_ref_pic(self.jpeg_path),sample_one_folder)
        ff_info = random.sample(self.get_degrade_ref_pic(self.ff_path),sample_one_folder)
        gblur_info = random.sample(self.get_degrade_ref_pic(self.gblur_path),sample_one_folder)
        wn_info = random.sample(self.get_degrade_ref_pic(self.wn_path),sample_one_folder)

        result = []
        result.extend(jp2k_info)
        result.extend(jpeg_info)
        result.extend(ff_info)
        result.extend(gblur_info)
        result.extend(wn_info)
        return result




if __name__ == '__main__':
    a = live_database2()
    data = a.get_degrade_ref_pic(a.gblur_path)
    #data = a.get_picinfo_all_folder(10)
    for item in data:
        print item





