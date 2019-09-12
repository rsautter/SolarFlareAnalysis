import Image as myImg
from PIL import Image
import numpy as np
cimport cython


# Reads in grayscale
cpdef read(filename):
    mat = np.asarray(Image.open(filename)).astype(np.float32)
    return myImg.Image(mat[:,:,0]),myImg.Image(mat[:,:,1]),myImg.Image(mat[:,:,2])

# normalizing and rastering
# It considers a single channel!
cpdef write(filename, img):
    mat = np.array(img.getMat())
    min,max = np.min(mat), np.max(mat)
    nmat = (mat-min) / (max-min)
    nmat = (nmat*255.0).astype(np.uint8)
    img = Image.fromarray(nmat,"L")
    img.save(filename)