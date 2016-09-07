import numpy as np
import cv2
from matplotlib import pyplot as plt
from LogGabor import LogGabor


lg = LogGabor("default_param.py")



exit(0)
lg.set_size(image)
fig_width = 512
phi = (np.sqrt(5) +1.)/2. # golden number
fig = plt.figure(figsize=(fig_width, fig_width/phi))
xmin, ymin, size = 0, 0, 1.
for i_level in range(8):
    a = fig.add_axes((xmin/phi, ymin, size/phi, size), axisbg='w')
    a.axis(c='b', lw=0)
    plt.setp(a, xticks=[])
    plt.setp(a, yticks=[])
    im_RGB = np.zeros((lg.N_X, lg.N_Y, 3))
    for theta in np.linspace(0, np.pi, 8, endpoint=False):
        params = {'sf_0':1./(2**i_level), 'B_sf':lg.pe.B_sf, 'theta':theta, 'B_theta':lg.pe.B_theta}
        # loggabor takes as args: u, v, sf_0, B_sf, theta, B_theta)
        FT_lg = lg.loggabor(0, 0, **params)
        im_abs = np.absolute(lg.FTfilter(image, FT_lg, full=True))
        RGB = np.array([.5*np.sin(2*theta + 2*i*np.pi/3)+.5 for i in range(3)])
        im_RGB += im_abs[:,:, np.newaxis] * RGB[np.newaxis, np.newaxis, :]

    im_RGB /= im_RGB.max()
    a.imshow(1-im_RGB, **opts)
    a.grid(False)
    i_orientation = np.mod(i_level, 4)
    if i_orientation==0:
        xmin += size
        ymin += size/phi**2
    elif i_orientation==1:
        xmin += size/phi**2
        ymin += -size/phi
    elif i_orientation==2:
        xmin += -size/phi
    elif i_orientation==3:
        ymin += size
    size /= phi
    #print i_orientation, xmin, ymin, size
if not(figpath==''):
    for ext in FORMATS: fig.savefig(figpath + 'fig_log_gabor_filters_B.' + ext)