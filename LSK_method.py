import cv2
from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg as la
import math
import time



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

def convPoint2Str(point):
    text = str(point[0])+'_'+str(point[1])
    return text


class LSK_method:
    def __init__(self,imgpath):
        self.pic = cv2.imread(imgpath,cv2.CV_LOAD_IMAGE_GRAYSCALE)
        self.pic_deri_x =  cv2.Sobel(self.pic, cv2.CV_32F, 1, 0)/8
        self.pic_deri_y =  cv2.Sobel(self.pic, cv2.CV_32F, 0, 1)/8
        self.h =1

        self.duration_getW = 0
        self.duration_getFmat = 0
        self.duration_getSaliency = 0

        self.matC_dict = {}
        self.K_dict = {}
        self.W_dict = {}


        self.Kcount_dirt = 0
        self.Kcount = 0

    def get_mat_C(self,center_p):
        roi_x = center_p[0]
        roi_y = center_p[1]
        block_deri_x = copyBlock_center(self.pic_deri_x,(roi_x,roi_y),5)
        block_deri_y = copyBlock_center(self.pic_deri_y,(roi_x,roi_y),5)
        roixarr = mat2array(block_deri_x)
        roiyarr = mat2array(block_deri_y)
        mat_j = np.hstack((roiyarr,roixarr))
        #mat_j = np.hstack((roixarr, roiyarr))
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







    def get_K(self,center_p,cur_p):
        K_key = convPoint2Str(center_p) + '_' + convPoint2Str(cur_p)
        self.Kcount +=1
        if K_key in self.K_dict.keys():
            K = self.K_dict[K_key]
            self.Kcount_dirt+=1
            return K
        else:
            center_p_key = convPoint2Str(center_p)
            if center_p_key not in self.matC_dict.keys():
                C = np.matrix(self.get_mat_C(center_p))
                self.matC_dict[center_p_key] = C
            else:
                C = self.matC_dict[center_p_key]



            K = (np.sqrt(la.det(C)) / (self.h ** 2)) * np.exp \
                ((np.matrix(center_p - cur_p) * C * np.matrix(center_p - cur_p).T) / (-2 * self.h ** 2))
            self.K_dict[K_key] = K
            return K

    def get_W(self,center_p,cur_p):
        w_key = convPoint2Str(center_p)+'_'+convPoint2Str(cur_p)
        if w_key in self.W_dict.keys():
            W_cur = self.W_dict[w_key]
        else:
            K_surround = 0
            for surround_p in get_surround_pixel(center_p):
                K_surround += self.get_K(surround_p,cur_p)
            K_center = self.get_K(center_p, cur_p)
            W_cur = K_center / K_surround
            self.W_dict[w_key] = W_cur
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
                K = self.get_K(center_p,cur_p)
                k_img[cur_x + filter_r][cur_y + filter_r] = K
        return k_img

    def get_W_img(self,center_p,size):
        begin = time.time()
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
        duration = time.time()-begin
        self.duration_getW += duration
        return w_img

    #P: size of LSK (in edge length)
    #L: number of LSK in feature matrix (count in edge length)
    def get_FeatureMatrix(self,center_p, P, L):
        begin = time.time()
        w_img = self.get_W_img(center_p, L + 2)
        F_matrix = np.zeros((P ** 2, L ** 2))
        count = 0
        for x in range(L):
            for y in range(L):
                cur_p = np.array([x + 1, y + 1])
                wblock = copyBlock_center(w_img, cur_p, P).reshape((P ** 2))
                F_matrix[:, count] = wblock
                count += 1
        #print 'finish get feature mat'
        duration = time.time()-begin
        self.duration_getFmat += duration
        return F_matrix

    # N: size of region for compution self-resemblance
    # P: size of LSK (in edge length)
    # L: number of LSK in feature matrix (count in edge length)
    def get_Saliency(self,center_p,N=7,P=3,L=3):
        begin = time.time()
        F_mat_list = []
        r = (N-1)/2
        for cur_x in range(N):
            cur_x -= r
            for cur_y in range(N):
                cur_y -= r
                delta_p = np.array([cur_x,cur_y])

                cur_p = center_p+delta_p
                F_mat = self.get_FeatureMatrix(cur_p,P,L)
                F_mat_list.append(F_mat)
        F_mat_center = self.get_FeatureMatrix(center_p,P,L)
        mat_similar = 0
        for mat in F_mat_list:
            mat_similar+=self.matrix_similar(mat,F_mat_center)
        duration = time.time() - begin
        self.duration_getSaliency += duration
        return 1/mat_similar


    def matrix_similar(self,mat1,mat2):
        mat1 = np.matrix(mat1)
        mat2 = np.matrix(mat2)

        if mat1.shape != mat2.shape:
            raise NameError('Matrix must have same shape')
        mat1norm = la.norm(mat1,'fro')
        mat2norm = la.norm(mat2,'fro')
        mat1 = mat1/mat1norm
        mat2 = mat2/mat2norm
        sigma = 0.07
        matDiffNorm = la.norm(mat1-mat2) ** 2
        similar = np.exp( -1 * matDiffNorm / (2 * sigma**2)  )
        return similar


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
        mat_j = np.hstack((roiyarr,roixarr))
        u, sigma, v = la.svd(mat_j)
        s1 = sigma[0]
        s2 = sigma[1]
        v1 = np.matrix(v[0,:])
        v2 = np.matrix(v[1, :])

        print 'pattern degree:',"{:.2f}".format(math.atan(v2[0,1]/v2[0,0])/math.pi *180)
        print 'V mat:\n',v
        a1 = (s1 + 1) / (s2 + 1)
        a2 = (s2 + 1) / (s1 + 1)
        gamma = np.power((s1 + s2 + 1e-6) / 9, 0.008)
        mat_c = gamma * ((a1 ** 2) * v1.T * v1 + (a2 ** 2) * v2.T * v2)
        print 'C mat:\n',mat_c
        print 'det C:\n',la.det(mat_c)






if __name__ == "__main__":
    lsk = LSK_method('pics/Lenna.png')
    center_p = np.array([50,50])
    cur_p = np.array([50,51])
    k = lsk.get_K(center_p,cur_p)



