
from matplotlib import pyplot as plt

import QoE_algorithm as qa
from Live_database2 import live_database2
import math
import random
import numpy as np
from matplotlib.pyplot import cm
from sklearn.linear_model import LogisticRegression
from scipy.optimize import curve_fit



def func_adj(x,a,b):
    return a*x+b

if __name__ == "__main__":
    live_db2 = live_database2()
    madlist = []
    dmoslist = []

    imagedata = live_db2.get_picinfo_all_folder(50)



    funcset = []
    funcset.append(qa.get_PSNR)
   # funcset.append(qa.get_SSIM)
   # funcset.append(qa.get_Total_MAD)

    for func in funcset:
        resultdata = qa.get_MADlist_by_multiProcess(imagedata,func)
        funcname = func.__name__.split('_')[-1]

        drawdata = {}
        dmoslist = []
        vallist = []
        for item in resultdata:
            dmos = item[0]
            val = item[1]
            #print dmos,val,'-------------'
            if funcname == 'MAD' and val != 0:
                val = math.log(val)

            dmoslist.append(dmos)
            vallist.append(val)
            losstype = item[2]
            if losstype not in drawdata.keys():
                drawdata[losstype] = []
                drawdata[losstype].append([dmos,val])
            else:
                drawdata[losstype].append([dmos, val])

        #-----logistics regression----#

        # lrclf = LogisticRegression()
        # print type(vallist[0]),type(dmoslist[0])
        #lrclf.fit(vallist,dmoslist)
        #print len(vallist),len(dmoslist),type(vallist[0]),type(dmoslist[0])
        # plt.figure(0)
        # plt.plot(dmoslist,vallist,'o')

        popt, pcov = curve_fit(func_adj,np.array(dmoslist),np.array(vallist),(-1,1))
        print popt,pcov


        plt.figure(funcname)
        color = iter(cm.rainbow(np.linspace(0, 1, len(drawdata.keys()))))
        for key in drawdata.keys():
            data = np.array(drawdata[key])
            dmos = data[:,0]
            val = data[:, 1]

            c=next(color)
            plt.plot(dmos,val,'o',color=c,label = key)
        y=[]
        for x in range(100):
            y.append(func_adj(x,popt[0],popt[1]))
        plt.plot(range(100),y,'ro-')

        plt.legend(loc=1)

        plt.xlabel('d-mos')
        plt.ylabel(funcname)
    plt.show()

