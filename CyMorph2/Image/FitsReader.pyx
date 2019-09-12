import astropy.io.fits as fits
import numpy as np
cimport cython
import Image

cpdef read(filename):
    return Image(np.array(fits.open(filename)[0].data, np.float32()))

def write(filename, img):
    mat = np.array(img.getMat())
    hdu = fits.PrimaryHDU(mat)
    hdu.writeto(filename, clobber=True)