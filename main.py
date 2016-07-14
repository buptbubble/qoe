import cv2
import dtcwt
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
import math
import os

def get_DT_CWT_Level1_Coff(img_gray,resize=1):
    transform = dtcwt.Transform2d()
    img_t = transform.forward(img_gray, nlevels=1)
    subBandCount = img_t.highpasses[0].shape[2]
    subband_list = []
    for subband in range(subBandCount):
        subbandpic = np.uint8(np.abs(img_t.highpasses[0][:, :, subband]))
        subband_h = subbandpic.shape[0]
        subband_w = subbandpic.shape[1]
        img_resize = cv2.resize(subbandpic, (subband_h * 2, subband_w * 2))
        if resize:
            subband_list.append(img_resize)
        else:
            subband_list.append(subbandpic)
    return subband_list

def get_percei_lum(img):
    b=0
    k=0.02874
    gamma = 2.2
    img_lum = (b+k*img)**gamma
    #print img_lum
    img_perci_lum = img_lum**(1*1.0/3)
    return img_perci_lum

def do_DFT(img):

    img_dft = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
    img_re = img_dft[:,:,0]
    img_im = img_dft[:,:,1]
    img_amp = np.log(cv2.magnitude(img_re,img_im))
    dft_shift = np.fft.fftshift(img_amp)

    plt.subplot(1,2,1)
    plt.imshow(dft_shift,cmap='gray')
    plt.title("DFT")

    ishift_amp = np.fft.ifftshift(dft_shift)

    img_rc = cv2.idft(img_dft)
    imgback = cv2.magnitude(img_rc[:,:,0],img_rc[:,:,1])
    print "img_rc",img_rc.shape
    plt.subplot(1,2,2)
    plt.imshow(imgback,cmap='gray')
    plt.title("Reconstruction")
    plt.show()



def get_freq_dirc(u,v,M=512,N=512,rho=150,theta=20):
    freq = ((u/(M*1.0/2))**2+(v/(N*1.0/2))**2)**0.5*(rho*theta*math.tan(math.pi/180))*0.5
    pha = math.atan(v*1.0/u)
    return freq,pha

def freq_trans_func(freq,theta):
    freq_theta = freq/(0.15*math.cos(4*theta)+0.85)

    trans_value = 2.6*(0.0192+0.114*freq_theta)*math.exp(-(0.114*freq_theta)**1.1)
    return trans_value


def get_filtered_CSF_img(img_in):
    img_dft = cv2.dft(np.float32(img_in), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(img_dft)
    height = img_dft.shape[0]
    weight = img_dft.shape[1]
    M = weight/2
    N = height/2
    H_matrix = np.zeros((height,weight))


    for h_idx in range(height):
        for w_idx in range(weight):
            m = -M + w_idx + 0.5
            n = -N + h_idx + 0.5
            freq,theta = get_freq_dirc(m,n,weight,height)
            multiVal = freq_trans_func(freq,theta)
            H_matrix[h_idx][w_idx] = multiVal

    img_magi = cv2.magnitude(img_dft[:,:,0],img_dft[:,:,1])
    img_magi *= H_matrix
    img_phase = cv2.phase(img_dft[:,:,0],img_dft[:,:,1])


    img_re = img_magi*np.cos(img_phase)
    img_im = img_magi*(np.sin(img_phase))

    img_dft2 = np.dstack((img_re,img_im))

    imgback = cv2.idft(img_dft2)
    imgback = cv2.magnitude(imgback[:,:,0],imgback[:,:,1])

    return imgback



def get_contrast_by_brocks(img):
    weight = img.shape[1]
    height = img.shape[0]

    w_blocks = int(math.floor(weight/16))
    h_blocks = int(math.floor(height/16))
    contrast_list = []
    for w_idx in range((w_blocks-1)*4+1):
        for h_idx in range((h_blocks-1)*4+1):
            roi = img[w_idx*4:w_idx*4+16,h_idx*4:h_idx*4+16]
            roi_mean = np.mean(roi)
            roi_std = np.std(roi)
            roi_contrast = roi_std/roi_mean
            contrast_list.append(roi_contrast)
    contrast_list = np.array(contrast_list)
    return contrast_list


def get_Total_MAD(img_ori_f,img_err_f):
    width = img_ori_f.shape[1]
    height = img_ori_f.shape[0]

    w_blocks = int(math.floor(width / 16))
    h_blocks = int(math.floor(height / 16))
    mad_score = 0
    count = 0
    weight_img = []
    for w_idx in range((w_blocks - 1) * 4 + 1):
        weight_line = []
        for h_idx in range((h_blocks - 1) * 4 + 1):
            roi_ori = img_ori_f[w_idx * 4:w_idx * 4 + 16, h_idx * 4:h_idx * 4 + 16]
            roi_ori_mean = np.mean(roi_ori)
            roi_ori_std = np.std(roi_ori)
            roi_contrast_ori = roi_ori_std / roi_ori_mean

            roi_err = img_err_f[w_idx * 4:w_idx * 4 + 16, h_idx * 4:h_idx * 4 + 16]
            roi_err_mean = np.mean(roi_err)
            roi_err_std = np.std(roi_err)
            roi_contrast_err = roi_err_std / roi_ori_mean
            if roi_contrast_err == 0:
                print 'Seems no err exits. Exit.'
                exit(0)
            c_err_ln = math.log(roi_contrast_err)
            c_ori_ln = math.log(roi_contrast_ori)
            thd = -5
            weight_block = -1
            if c_err_ln>c_ori_ln and c_ori_ln>thd:
                weight_block = c_err_ln-c_ori_ln
            elif c_err_ln>thd and c_ori_ln<=thd:
                weight_block = c_err_ln - thd
            else:
                weight_block = 0
            weight_line.append(weight_block)

            D = np.sum(np.power(roi_err,2))/(16**2)
            D_weighted = D*weight_block
            mad_score+=D_weighted**2
            count+=1
        weight_img.append(weight_line)

    weight_img = np.array(weight_img)
    weight_img = cv2.resize(weight_img,img_ori_filtered.shape)
    plt.figure("Weight Image")
    plt.imshow(weight_img,cmap='gray')
    mad_score = (mad_score/count)**0.5


    return mad_score



if __name__ == "__main__":
    picpath = './pics'
    imgname_ori = 'barbara.png'
    imgname_de = 'barbara_dark2.jpg'
    #imgname_de = imgname_ori

    img_ori_path = os.path.join(picpath,imgname_ori)
    img_degraded_path = os.path.join(picpath,imgname_de)
    img_ori = cv2.imread(img_ori_path,cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img_de = cv2.imread(img_degraded_path,cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img_diff = img_ori-img_de




    img_ori_L = get_percei_lum(img_ori)
    img_ori_filtered = get_filtered_CSF_img(img_ori_L)

    img_error_L = get_percei_lum(img_ori)-get_percei_lum(img_de)
    img_error_filtered = get_filtered_CSF_img(img_error_L)

    # contrast_ori = get_contrast_by_brocks(img_ori_filtered)
    # contrast_diff = get_contrast_by_brocks(img_error_filtered)
    # print np.mean(contrast_ori), np.std(contrast_ori)
    # print np.mean(contrast_diff), np.std(contrast_diff)

    # plt.figure(0)
    # plt.subplot(1,3,1)
    # plt.imshow(img_ori,cmap='gray')
    # plt.subplot(1,3,2)
    # plt.imshow(img_de, cmap='gray')
    # plt.subplot(1, 3, 3)
    # plt.imshow(img_error_L, cmap='gray')





    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(img_ori_filtered,cmap='gray')
    plt.title("Lumia CSF filtered Origin Pic.")
    plt.subplot(1,2,2)
    plt.imshow(img_error_filtered,cmap='gray')
    plt.title("Lumia CSF filtered Error Pic.")

    score = get_Total_MAD(img_ori_filtered,img_error_filtered)
    print 'MAD score: ',score

    plt.show()


