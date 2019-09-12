from math import radians

from math import radians,sqrt,pow
from scipy.ndimage.interpolation import rotate
from scipy import fftpack
import numpy

class Image:

    def __init__(self,  mat):
        self.mat = mat

    # Abstract methods:
    def segment(self):
        pass

    def getCleaned(self):
        pass

    def getMat(self):
        return self.mat

    #Image methods
    def applyMask(self, mask):
        w, h = len(self.mat[0]), len(self.mat)
        mat = numpy.array([[float(0.0) for j in range(w)] for i in range(h)])
        for i in range(w):
            for j in range(h):
                if(mask[j, i] == 1.0):
                    mat[j, i] = 0.0
                else:
                    mat[j, i] = self.mat[j, i]
        return Image(mat,self.aimList)

    def removeBackground(self,background):

        w, h = len(self.mat[0]), len(self.mat)
        mat = numpy.array([[float(0.0) for j in range(w)] for i in range(h)])
        for i in range(w):
            for j in range(h):
                mat[j, i] = self.mat[j, i]-background
        return Image(mat)

    def _butterworth(self, d, d0, n):
        return 1.0/(1.0+pow(d/d0,2.0*n))

    def filterButterworth2D(self,  degradation,  n=2):
        # frac -> max smoothing at center
        
        img = self.getMat()
        heigth, width = len(img),len(img[0])

        smoothed = numpy.array([[0.0 for i in range(width)] for j in range(heigth)])
        zeros = numpy.array([[0.0 for i in range(width)] for j in range(heigth)])
        temp = numpy.array([[0.0 for i in range(width)] for j in range(heigth)])
        freq = fftpack.fft2(img)
        maxD0 = sqrt(pow(float(width),2.0)+pow(float(heigth),2.0))
        d0 = float(maxD0)*degradation
        newFreq = numpy.array([[freq[j][i] for i in range(width)] for j in range(heigth)], dtype=numpy.complex64)

        for i in range(heigth):
            for j in range(width):
                newFreq[i][j] = freq[i][j]*self._butterworth(float(sqrt(pow(i-heigth/2,2.0)+pow(j-width/2,2.0))),float(d0),float(n))
        smoothed = numpy.real(fftpack.ifft2(newFreq))
        return Image(smoothed)

    def gradient(self):
        w, h = len(self.mat[0]), len(self.mat)
        dx = numpy.array([[0.0 for i in range(w)] for j in range(h)])
        dy = numpy.array([[0.0 for i in range(w)] for j in range(h)])
        for y in range(h):
            for x in range(w):
                #dy gradient:
                dist = 2.0
                p1x, p1y = x, y
                p2x,p2y = x, y

                if (y-1<0):
                    dist = 1.0
                else:
                    p1y = y-1
                if (y+1 >= h):
                    dist = 1.0
                else:
                    p2y = y+1
                dy[y, x] = (self.mat[ p2y, p2x ] - self.mat[ p1y, p1x ])/dist

                #dx gradient:
                dist = 2.0
                p1x, p1y = x, y
                p2x,p2y = x, y
                if (x-1 < 0):
                    dist = 1.0
                else:
                   p1x = x-1
                if (x+1 >= w ):
                    dist = 1.0
                else:
                    p2x = x+1
                dx[y, x] = (self.mat[ p2y, p2x ] - self.mat[ p1y, p1x ])/dist
        return Image(dy),Image(dx)

    def rotateImage(self, angle):
        imgR = rotate(self.getMat(), angle, reshape=False,mode='nearest')
        return Image(imgR)



