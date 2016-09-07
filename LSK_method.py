import cv2
from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg as la



def windowPic(img,x,y,sizex,sizey):
    return img[y:y+sizey,x:x+sizex]

def mat2array(img):
    img_size = img.shape[0] * img.shape[1]
    return img.reshape(img_size,1)

def LSK(center_x, cur_x,C,h):

    det_c = la.det(mat_c)

    return (np.sqrt(det_c)/(h**2))*np.exp((np.matrix(center_x-cur_x)*C*np.matrix(center_x-cur_x).T)/(-2 * h**2))

def get_mat_C(roi_x,roi_y,pic_ori):
    pic_deri_y = cv2.Sobel(pic_ori, cv2.CV_32F, 0, 1)
    pic_deri_x = cv2.Sobel(pic_ori, cv2.CV_32F, 1, 0)
    roipic_derix = windowPic(pic_deri_x, roi_x, roi_y, 5, 5)
    roipic_deriy = windowPic(pic_deri_y, roi_x, roi_y, 5, 5)
    roixarr = mat2array(roipic_derix)
    roiyarr = mat2array(roipic_deriy)
    mat_j = np.hstack((roixarr, roiyarr))
    u, sigma, v = la.svd(mat_j)
    s1 = sigma[0]
    s2 = sigma[1]
    v1 = np.matrix(v[:, 0])
    v2 = np.matrix(v[:, 1])
    a1 = (s1 + 1) / (s2 + 1)
    a2 = (s2 + 1) / (s1 + 1)
    gamma = np.power((s1 + s2 + 1e-6) / 9, 0.008)

    mat_c = gamma * a1 ** a1 * v1.T * v1 + a2 ** a2 * v2.T * v2
    return mat_c

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


if __name__ == "__main__":
    pics = cv2.imread('pics/Lenna.png',cv2.CV_LOAD_IMAGE_GRAYSCALE)
    roi_x = 50
    roi_y = 60

    mat_c = get_mat_C(roi_x,roi_y,pics)
    center_x = 50
    center_y = 50
    filter_r = 5
    h=1
    center_arr = np.array([center_x,center_y])


    lsk_filter = np.zeros((filter_r*2,filter_r*2))

    for cur_x in range(filter_r*2):
        cur_x = cur_x-filter_r
        for cur_y in range(filter_r*2):
            cur_y = cur_y-filter_r

            center_p = np.array([center_x,center_y])
            cur_p = np.array([center_x+cur_x,center_y+cur_y])
            K_surround = 0
            for surround_p in get_surround_pixel(center_p):
                K_surround += LSK(cur_p,surround_p,mat_c,h)
            K_center = LSK(center_p,cur_p,mat_c,h)
            W_cur = K_center/K_surround
            lsk_filter[cur_x+filter_r][cur_y+filter_r] = W_cur
    plt.imshow(lsk_filter,cmap='gray',interpolation='none')
    plt.show()




    #LSK()

    print 'mat_c:\n',mat_c


    exit(0)