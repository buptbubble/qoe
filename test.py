import cv2
import numpy as np

def listtostr(alist):
    line = ''
    for item in alist:
        line=line + str(item)+','
    return line.strip(',')


fr = open('dmos.txt','r')
for line in fr:
    lineC = line.split(',')
    jp2k = lineC[0:227]
    jpeg = lineC[227:460]
    wn = lineC[460:634]
    gauss = lineC[634:808]
    ff = lineC[808:982]
    dmos = [jp2k,jpeg,wn,gauss,ff]


    fw = open('jp2k.txt','w')
    fw.write(listtostr(jp2k))
    fw.close()

    fw = open('jpeg.txt', 'w')
    fw.write(listtostr(jpeg))
    fw.close()

    fw = open('wn.txt', 'w')
    fw.write(listtostr(wn))
    fw.close()

    fw = open('gauss.txt', 'w')
    fw.write(listtostr(gauss))
    fw.close()

    fw = open('ff.txt', 'w')
    fw.write(listtostr(ff))
    fw.close()

    print len(lineC)
    break
