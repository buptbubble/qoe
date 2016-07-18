import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
from scipy.optimize import curve_fit


def func(x,a,b):
    return a*x+b

x=range(10)
y = [func(xx,-50,10)+random.random() for xx in x]
para = curve_fit(func,x,y)
print para
plt.plot(x,y,'ro-')
plt.show()