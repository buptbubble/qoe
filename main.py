
from matplotlib import pyplot as plt

import QoE_algorithm as qa
from Live_database2 import live_database2
import math
import random
import numpy as np

if __name__ == "__main__":
    live_db2 = live_database2()
    madlist = []
    dmoslist = []

    imagedata_jp2k = random.sample(live_db2.get_degrade_ref_pic(live_db2.jp2k_path),50)
    imagedata_jp2k = random.sample(live_db2.get_degrade_ref_pic(live_db2.jp2k_path),50)
    imagedata_jp2k = random.sample(live_db2.get_degrade_ref_pic(live_db2.jp2k_path),50)
    imagedata_jp2k = random.sample(live_db2.get_degrade_ref_pic(live_db2.jp2k_path),50)


    # for img in imagedata:
    #     imgori = img[0]
    #     imgde = img[1]
    #     ssim = qoe.getSSIM(imgori,imgde)
    #     psnr = qoe.getPSNR(imgori,imgde)
    #     print 'ssim:',ssim,'\tpsnr:',psnr
    #
    #
    # exit(0)
    funcset = []
    funcset.append(qa.get_PSNR)
    funcset.append(qa.get_SSIM)
    funcset.append(qa.get_Total_MAD)

    for func in funcset:
        resultdata = qa.get_MADlist_by_multiProcess(imagedata,func)
        funcname = func.__name__.split('_')[-1]
        resultdata = np.array(resultdata)

        dmos = resultdata[:,0]

        val = resultdata[:,1]
        if funcname == 'MAD':
            for i,item in enumerate(val):
                if item!=0:
                    val[i] = np.log(val[i])
        plt.figure(funcname)
        plt.plot(dmos,val,'o')
        plt.title('Total samples:'+str(dmos.size))
        plt.xlabel('d-mos')
        plt.ylabel(funcname)
    plt.show()

