import numpy as np

isize = 1024
rsun = 392
binning = 4
isize_new = isize//binning

## circle filter to consider only solar disk.
X = np.arange(isize_new)[:, None]
Y = np.arange(isize_new)[None, :]
XY = np.sqrt((X-isize_new/2.)**2. + (Y-isize_new/2.)**2.)
cfilter = np.where(XY<rsun)

## resize
from skimage.transform import resize
def resize_(img_, shape_):
    return resize(img, (shape_, shape_), order=1, mode='constant', preserve_range=True)

## resize and flatten
def resize_and_flatten(img_, isize_new, cfilter):
    img_ = resize_(img_, isize_new)
    arr_ = img_[Z]
    arr_ = ((arr_+1.).clip(1, 2.**14.)).flatten()
    ## +1. is to avoid zero division error when calculate PPE10 and PPE50
    return arr_
    
## calculate CC, RE, PPE10, PPE50
def cal_scores(file_tar, file_gen, isize_new, cfilter):

    img_tar = np.load(file_tar)
    img_gen = np.load(file_gen)
    arr_tar = resize_and_flatten(img_tar, isize_new, cfilter)
    arr_gen = resize_and_flatten(img_gen, isize_new, cfilter)

    cc = np.corrcoef(arr_tar, arr_gen)[0, 1]
    flux_tar = np.sum(arr_tar)
    flux_gen = np.sum(arr_gen)
    re = (flux_gen-flux_tar)/flux(tar)
    ppe = np.abs(arr_gen-arr_tar)/arr_tar
    ppe10 = ((np.where(ppe <= 0.1)[0].shape)[0])/arr_tar.shape[0]
    ppe50 = ((np.where(ppe <= 0.5)[0].shape)[0])/arr_tar.shape[0]

    return cc, re, ppe10, ppe50

