import cv2
from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg as la
import math


def copyBlock_center(img,center,size):
    if size%2 != 1:
        raise NameError('Image block size error')
    x = center[0]
    y = center[1]
    r = (size-1)/2
    upper = x-r
    bottom = x+r+1
    left = y-r
    right = y+r+1
    if upper<0:
        upper=0
    if bottom>img.shape[0]:
        bottom=img.shape[0]
    if left<0:
        left=0
    if right>img.shape[1]:
        right=img.shape[1]
    return img[upper:bottom,left:right]

def mat2array(img):
    img_size = img.shape[0] * img.shape[1]
    return img.reshape(img_size,1)

def get_surround_pixel(center_p):
    center_x = center_p[0]
    center_y = center_p[1]
    if center_x == 0 or center_y == 0:
        yield 1
    for curx in range(3):
        curx -=1
        for cury in range(3):
            cury -=1

            curp = np.array([curx+center_x,cury+center_y])
            if not (curp == center_p).all():
                yield curp
            else:
                continue


class LSK_method:
    def __init__(self,imgpath):
        self.pic = cv2.imread(imgpath,cv2.CV_LOAD_IMAGE_GRAYSCALE)
        self.pic_deri_x =  cv2.Sobel(self.pic, cv2.CV_32F, 1, 0)/8
        self.pic_deri_y =  cv2.Sobel(self.pic, cv2.CV_32F, 0, 1)/8
        self.h =1

    def get_mat_C(self,center_p):
        roi_x = center_p[0]
        roi_y = center_p[1]
        block_deri_x = copyBlock_center(self.pic_deri_x,(roi_x,roi_y),5)
        block_deri_y = copyBlock_center(self.pic_deri_y,(roi_x,roi_y),5)
        roixarr = mat2array(block_deri_x)
        roiyarr = mat2array(block_deri_y)
        mat_j = np.hstack((roixarr, roiyarr))
        u, sigma, v = la.svd(mat_j)
        s1 = sigma[0]
        s2 = sigma[1]
        v1 = np.matrix(v[0, :])
        v2 = np.matrix(v[1, :])
        a1 = (s1 + 1) / (s2 + 1)
        a2 = (s2 + 1) / (s1 + 1)
        gamma = np.power((s1 + s2 + 1e-6) / 9, 0.008)

        mat_c = gamma * ((a1 ** 2) * v1.T * v1 + (a2 ** 2) * v2.T * v2)
        return mat_c

    def get_mat_C_byJ(self,center_p):
        roi_x = center_p[0]
        roi_y = center_p[1]
        block_deri_x = copyBlock_center(self.pic_deri_x, (roi_x, roi_y), 5)
        block_deri_y = copyBlock_center(self.pic_deri_y, (roi_x, roi_y), 5)
        roixarr = mat2array(block_deri_x)
        roiyarr = mat2array(block_deri_y)
        mat_j = np.matrix(np.hstack((roixarr, roiyarr)))
        C = mat_j.T * mat_j
        return C


    def get_K_info(self,center_p,cur_p):
        C = np.matrix(self.get_mat_C(center_p))
        c_multi = np.matrix(center_p - cur_p) * C * np.matrix(center_p - cur_p).T
        print 'c_multi:',c_multi
        expval = math.exp(c_multi/-2)
        print 'expval:',expval



    def get_K(self,center_p,cur_p):
        C = np.matrix(self.get_mat_C(center_p))
        det_c = la.det(C)
        return (np.sqrt(det_c) / (self.h ** 2)) * np.exp\
            ((np.matrix(center_p - cur_p) * C * np.matrix(center_p - cur_p).T) / (-2 * self.h ** 2))

    def get_W(self,center_p,cur_p):
        K_surround = 0
        for surround_p in get_surround_pixel(center_p):
            #K_surround += self.get_K(surround_p,cur_p)
            K_surround += self.get_K(surround_p,cur_p)

        K_center = self.get_K(center_p, cur_p)
        W_cur = K_center / K_surround
        return W_cur

    def get_K_img(self, center_p, size):
        k_img = np.zeros((size, size))

        if size % 2 != 1:
            raise NameError('size must be odder')
        filter_r = (size - 1) / 2
        for cur_x in range(size):
            cur_x = cur_x - filter_r
            for cur_y in range(size):
                cur_y = cur_y - filter_r
                delta_p = np.array([cur_x, cur_y])

                cur_p = center_p + delta_p
                K = self.get_K(center_p, cur_p)
                k_img[cur_x + filter_r][cur_y + filter_r] = K
        return k_img

    def get_W_img(self,center_p,size):
        w_img = np.zeros((size,size))

        if size%2 != 1:
            raise NameError('size must be odder')
        filter_r = (size-1)/2
        for cur_x in range(size):
            cur_x = cur_x - filter_r
            for cur_y in range(size):
                cur_y = cur_y - filter_r
                delta_p = np.array([cur_x,cur_y])

                cur_p = center_p + delta_p
                W = self.get_W(center_p, cur_p)
                w_img[cur_x + filter_r][cur_y + filter_r] = W
        return w_img

    def getROI(self,center_p,size):
        picblock = copyBlock_center(self.pic,center_p,size)
        return picblock

    def getDeriX(self,center_p,size):
        picblock = copyBlock_center(self.pic_deri_x,center_p,size)
        return picblock

    def getDeriY(self,center_p,size):
        picblock = copyBlock_center(self.pic_deri_y,center_p,size)
        return picblock


    def printInfo(self,center_p):
        roi_x = center_p[0]
        roi_y = center_p[1]
        block_deri_x = copyBlock_center(self.pic_deri_x, (roi_x, roi_y), 5)
        block_deri_y = copyBlock_center(self.pic_deri_y, (roi_x, roi_y), 5)
        roixarr = mat2array(block_deri_x)
        roiyarr = mat2array(block_deri_y)
        mat_j = np.hstack((roixarr, roiyarr))
        u, sigma, v = la.svd(mat_j)
        s1 = sigma[0]
        s2 = sigma[1]
        v1 = np.matrix(v[0,:])
        v2 = np.matrix(v[1, :])
        #print 'v1:',v1
        #print 'v2:',v2
        print 'pattern degree:',"{:.2f}".format(math.atan(v2[0,0]/v2[0,1])/math.pi *180)

        print 'V mat:\n',v
        #print 'Sigma mat:\n',sigma
        a1 = (s1 + 1) / (s2 + 1)
        a2 = (s2 + 1) / (s1 + 1)
        gamma = np.power((s1 + s2 + 1e-6) / 9, 0.008)
        #print 'gamma:',gamma,' a1_2:',a1,'  a2_2:',a2
        #print 'v1.T * v1\n',(a1 ** 2) * v1.T * v1
        #print 'v2.T * v2\n',(a2 ** 2) * v2.T * v2
        mat_c = gamma * ((a1 ** 2) * v1.T * v1 + (a2 ** 2) * v2.T * v2)
        print 'C mat:\n',mat_c
        print 'det C:\n',la.det(mat_c)






if __name__ == "__main__":
    lsk = LSK_method('pics/Lenna.png')








    plt.imshow(wimg, cmap='gray', interpolation='none')
    plt.show()
