from matplotlib import pyplot as plt
import QoE_algorithm as qa
from Live_database2 import live_database2
import math
import random
import numpy as np
from matplotlib.pyplot import cm
from sklearn.linear_model import LogisticRegression
from scipy.optimize import curve_fit


if __name__ == '__main__':
    live_db2 = live_database2()
    madlist = []
    dmoslist = []
    imagedata = live_db2.get_degrade_ref_pic(live_db2.jp2k_path)
    for data in imagedata:
        imgref = data[0]
        imgde = data[1]
        mad = qa.get_Total_MAD(imgref,imgde)
        print mad,imgref