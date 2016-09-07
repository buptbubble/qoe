# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import QoE_algorithm as qa
from Live_database2 import live_database2
import math
import random
import numpy as np
from matplotlib.pyplot import cm
from scipy.optimize import curve_fit



def func_adj(x,a,b,c,d):
    return a/(1+np.exp((x*1.0-d)/b))+c

if __name__ == "__main__":
    live_db2 = live_database2()
    madlist = []
    dmoslist = []

    imagedata = live_db2.get_picinfo_all_folder(10)
    #imagedata = live_db2.get_degrade_ref_pic(live_db2.wn_path)

    funcset = []
    #funcset.append(qa.get_PSNR)
    funcset.append(qa.get_SSIM)
    #funcset.append(qa.get_Total_MAD)
    funcset.append(qa.getMAD)
    funcset.append(qa.getWeightedMAD)

    for func in funcset:
        resultdata = qa.get_MADlist_by_multiProcess(imagedata,func,4)

        #print resultdata
        funcname = func.__name__.split('_')[-1]

        drawdata = {}
        dmoslist = []
        vallist = []
        for item in resultdata:
            dmos = item[0]
            val = item[1]
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

        popt, pcov = curve_fit(func_adj,vallist,dmoslist,p0=(28,15,18,28),maxfev=5000)
        #print popt

        #----calculate correlation coeff---------#
        dmoslist = np.array(dmoslist)
        vallist_adj = np.array([func_adj(x,popt[0],popt[1],popt[2],popt[3]) for x in vallist])
        corcoeff = np.corrcoef(dmoslist,vallist_adj,rowvar=0)[0][1]

        #--------drawing------------#
        f_adj = plt.figure(funcname)
        f_ori = plt.figure(funcname+' Origin')
        ax = f_adj.gca()
        bx = f_ori.gca()
        color = iter(cm.rainbow(np.linspace(0, 1, len(drawdata.keys()))))

        valmin = min(vallist)
        valmax = max(vallist)
        vallen = len(vallist)

        for key in drawdata.keys():
            data = np.array(drawdata[key])
            dmos = data[:,0]
            val = data[:, 1]
            val_adj = []
            for val_temp in val:
                val_a = func_adj(val_temp, popt[0], popt[1], popt[2], popt[3])
                val_adj.append(val_a)
            c=next(color)
            plt.sca(ax)
            plt.plot(dmos,val_adj,'o',color=c,label = key)
            plt.sca(bx)
            plt.plot(dmos, val, 'o', color=c, label=key)

        valcurve = np.linspace(valmin,valmax,vallen)
        dmoscurve = np.array([func_adj(xx, popt[0], popt[1], popt[2], popt[3]) for xx in valcurve])
        plt.sca(bx)
        plt.plot(dmoscurve,valcurve,'r--')
        plt.xlabel('d-mos')
        plt.ylabel(funcname)
        plt.legend(loc=4)

        plt.sca(ax)
        plt.plot([0,100],[0,100],ls = '-',color = 'r')
        plt.legend(loc=2)
        plt.title("Pearson Coeff.: "+"{:.3f}.".format(corcoeff))
        plt.xlabel('d-mos')
        plt.ylabel(funcname+" (Adjusted)")
        plt.ylim((0,100))
        plt.xlim((0,100))
    plt.show()

