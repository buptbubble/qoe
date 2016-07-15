
from matplotlib import pyplot as plt

from QoE_algorithm import QoE_algorithm
from Live_database2 import live_database2
import math

if __name__ == "__main__":
    qoe = QoE_algorithm()
    live_db2 = live_database2()
    madlist = []
    dmoslist = []

    imagedata = live_db2.get_degrade_ref_pic(live_db2.jp2k_path)
    for info in imagedata:
        imgref = info[0]
        imgde = info[1]
        dmos = info[2]
        mad = qoe.get_Total_MAD(imgref,imgde)
        print 'MAD log score: ', mad, "  DMOS:", dmos
        if dmos != 0:
            if mad == 0:
                mad = 10
            madlog = math.log(mad)
            madlist.append(madlog)
            dmoslist.append(dmos)

    # for imgref, imgde, dmos in live_db2.get_degrade_ref_pic(live_db2.jp2k_path):
    #     mad = qoe.get_Total_MAD(imgref, imgde)
    #     print 'MAD log score: ', mad,"  DMOS:",dmos
    #     if dmos != 0:
    #         if mad == 0:
    #             mad = 10
    #         madlog = math.log(mad)
    #         madlist.append(madlog)
    #         dmoslist.append(dmos)
    plt.plot(madlog,dmos,ls='or')
    plt.xlabel('log-MAD')
    plt.ylabel('DMOS')
    plt.show()


