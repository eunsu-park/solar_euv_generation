import numpy as np
from skimage.transform import resize

def rescale(data, imin, imax, omin, omax):
    odif = omax-omin
    idif = imax-imin
    data = (data-imin)*(odif/idif) + omin
    return data.clip(omin, omax)

def bytescale(data, imin=None, imax=None):
    if not imin:
        imin = np.min(data)
    if not imax:
        imax = np.max(data)
    data = rescale(data, imin, imax, omin=0, omax=255)
    return data.astype(np.uint8)

# Maybe almost same function to aia_intscale.pro in SSW.
class aia_intscale():
    list_aia = [94, 131, 171, 193, 211, 304, 335, 1600, 1700, 4500]
    def __init__(self, wavelnth):
        self.wavelnth = str(int(wavelnth))
        if int(self.wavelnth) not in list_aia:
            raise ValueError('%d is invalid AIA wavelength'%int(self.wavelnth))
    def aia_rescale(self, data):
        if self.wavelnth == '94':
            data = np.sqrt((data*4.99803).clip(1.5, 50.))
        elif self.wavelnth == '131':
            data = np.log10((data*6.99685).clip(7.0, 1200.))
        elif self.wavelnth == '171':
            data = np.sqrt((data*4.99803).clip(10., 6000.))
        elif self.wavelnth == '193':
            data = np.log10((data*2.99950).clip(120., 6000.))
        elif self.wavelnth == '211':
            data = np.log10((data*4.99801).clip(30., 13000.))
        elif self.wavelnth == '304':
            data = np.log10((data*4.99941).clip(50., 2000.))
        elif self.wavelnth == '335':
            data = np.log10((data*6.99734).clip(3.5, 1000.))
        elif self.wavelnth == '1600':
            data = (data*2.99911).clip(0., 1000.))
        elif self.wavelnth == '1700':
            data = (data*1.00026).clip(0., 2500.))
        elif self.wavelnth == '4500':
            data = (data*1.00026).clip(0., 26000.))
        data = bytescale(data)
    def __call__(self, data):
        data = self.aia_rescale(data)
        return data

class make_tensor():
    def __init__(self, isize, is_aia):
        self.is_aia = is_aia
        self.isize = isize
    def resize_tensor(self, x):
        if x.shape != (self.isize, self.isize) :
            x = resize(x, (self.isize, self.isize), order=1, mode='constant', preserve_range=True)
        x.shape = (1, self.isize, self.isize, 1)
        return x
    def __call__(self, x):
        x = np.load(x)
        if self.is_aia :
            x = np.log2((x+1.).clip(1., 2.**14.))
            x = rescale(x, imin=0., imax=14., omin=-1, omax=1)
        else :
            x = x.clip(-3000., 3000.)
            x = rescale(x, imin=-3000., imax=3000., omin=-1, omax=1)
        x = self.resize_tensor(x)
        return x.astype(np.float32)
    
class shake_tensor():
    def __init__(self, isize):
        self.isize = isize
    def pad(self, data_A, data_B):
        pad = int(self.isize/64) - 1
        x, y = np.random.randint(2*pad+1), np.random.randint(2*pad+1)
        data_A = np.pad(data_A, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
        data_B = np.pad(data_B, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
        data_A = data_A[:, x:x+self.isize, y:y+self.isize, :]
        data_B = data_B[:, x:x+self.isize, y:y+self.isize, :]
        return data_A, data_B
    def __call__(self, data_A, data_B):
        data_A, data_B = self.pad(data_A, data_B)
        return data_A.astype(np.float32), data_B.astype(np.float32)
    
class make_output():
    def __init__(self, isize, wavelnth):
        self.isize = isize
        self.intscale = aia_intscale(wavelnth)
    def dodo(self, result):
        result_npy = rescale(result, imin=-1., imax=1., omin=0., omax=14.)
        result_npy = 2.**result_npy - 1.
        result_png = self.intscale(result_npy)
        return result_npy, result_png
    def __call__(self, x):
        x.shape = (self.isize, self.isize)
        x_npy, x_png = dodo(x)
        return x_npy, x_png
    
    
