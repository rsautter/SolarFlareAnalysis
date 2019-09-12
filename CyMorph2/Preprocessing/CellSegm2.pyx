import cv2
import numpy as np
from Image import Image
import JpegReader
import matplotlib.pyplot as plt
from random import sample
from scipy.interpolate import interp2d

class CellSegm(object):

    def __init__(self,xtraID=0):
        self.__xtraID = xtraID

    def __clean(self):
        cleaned = np.array(self.__img.getMat().copy())
        original = np.array(self.__img.getMat())
        background = cleaned[self.__background==1]
        px, py = np.where(self.__mask==0)
        self._backAvg, self._backStd = np.average(background), np.std(background)
        w = 0.1
        cleaned[px,py] = w*original[px,py]+(1.0-w)*np.random.normal(self._backAvg, self._backStd,len(cleaned[px,py]))
        self.__cleaned = cleaned

    def circleTransform(self, pts,nrand=3000):
        cdef:
            int dx, dy,i, j, npoints
            int[:,:] transformada
            float ldist, dist
            long[:] p, p2

        points = sample(np.transpose(pts),nrand)
        dist = float(((max(pts[0])-min(pts[0]))**2.0+(max(pts[0])-min(pts[1]))**2.0)**0.5)
        dists = range(int(dist))
        transformada = np.zeros((len(points),len(dists))).astype(np.int32)
        npoints = len(points)
        for dx in range(npoints):
            p = points[dx]
            for i in range(npoints):
                p2 = points[i]
                ldist = float(((p2[0]-p[0])**2.0+(p2[1]-p[1])**2.0)**0.5)
                dy = int(ldist)
                transformada[dx][dy] = transformada[dx][dy] + 1
        densidade = np.array(transformada).astype(np.float)
        for i in dists:
            if(i == 0):
                continue
            densidade[:,i] = (densidade[:,i])
        maxAmp = np.unravel_index(np.argmax(densidade), (npoints,len(dists)))

        return points[maxAmp[0]], dists[maxAmp[1]]


    def preprocessImage(self, img):
        mask = np.zeros(img.getMat().shape).astype(np.float)
        nsamples = 30
        for i in np.linspace(0.2,0.8,nsamples):
           _, lmask, _ = self.segmentImage(img,i)
           mask += np.array(lmask.getMat()).astype(np.float)

        mask = mask/float(nsamples)

        self.__mask = np.zeros(mask.shape).astype(np.uint8)
        self.__mask[mask>=0.3] = 1
        self.__cleaned = np.array(self.__img.getMat().copy())
        self.__cleaned[self.__mask < 0.5] = np.random.normal(self._backAvg, self._backStd,len(self.__cleaned[self.__mask < 0.5]))
        self.__cleaned = cv2.blur(self.__cleaned,(3,3))

        return Image(self.__cleaned.astype(np.float32)), Image(self.__mask.astype(np.float32)), Image(self.__mask.astype(np.float32))


    def segmentImage(self, img, distThr=0.2):
        imgDimY, imgDimX = img.getMat().shape
        self.__img = img
        conv = np.array([
                [0,0,1,0,0],
                [0,1,1,1,0],
                [1,1,1,1,1],
                [0,1,1,1,0],
                [0,0,1,0,0]
                ]).astype(np.uint8)


        max = np.max(img.getMat())
        min = np.min(img.getMat())
        newMat = np.array(img.getMat())
        newMat = (255*(newMat-min)/(max-min)).astype(np.uint8)


        #Iniciando limpeza...

        isLessThan = np.where(cv2.blur(newMat,(15,15))<110)

        x, y = isLessThan
        rnewMat = np.zeros(newMat.shape)
        rnewMat[isLessThan] = 1
        ret, rnewMat = cv2.connectedComponents(rnewMat.astype(np.uint8))

        for blob in np.unique(rnewMat):
            if blob ==0:
                continue
            x, y = np.where(rnewMat == blob)
            minX, maxX = np.min(x)-2,np.max(x)+2
            minY, maxY = np.min(y)-2,np.max(x)+2
            rx, ry = np.where(rnewMat != blob)
            arg = np.where((rx>=minX) & (rx<=maxX))
            rx, ry = rx[arg], ry[arg]
            arg = np.where((ry>=minY) & (ry<=maxY))
            rx,ry = rx[arg], ry[arg]
            z = np.array(newMat[rx,ry])
            if len(z)>1:
                newMat[x, y] = np.random.normal(np.average(z),np.sqrt(np.std(z)),len(x))

        # Detecting objects and background:
        ret, thresh = cv2.threshold(newMat,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  conv,iterations=2)
        dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE,  conv)

        distBlob = cv2.distanceTransform(open,cv2.DIST_L2,5)

        ret, foreground = cv2.threshold(distBlob,distThr*distBlob.max(),255,0)

        foreground = foreground.astype(np.uint8)
        ret, markers = cv2.connectedComponents(foreground)
        markers = markers+1
        unknown = cv2.subtract(dilate,foreground)

        markers[unknown>250] = 0
        markers = cv2.watershed(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR),markers)
        markers = markers-1


        blobs = np.unique(markers)
        adists = []

        # Tracking each cell
        for blob in blobs:
            if blob < 1:
                adists.append(np.infty)
                continue # background
            elif len(np.where(markers == blob)[0])< 17:
                adists.append(np.infty)
                continue # background
            else:
                pts = np.where(markers == blob)
                dists = ((pts[0]-imgDimY/2.0)**2.0+ (pts[1]-imgDimX/2.0)**2.0)**(0.5)
                adists.append(np.average(dists))

        centralBlob = blobs[np.argmin(adists)]



        mask = markers.copy()
        toClean = markers.copy()
        background = markers.copy()

        mask[markers != centralBlob] = 0
        mask[markers == centralBlob] = 1
        self.__mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,  conv,iterations=2)

        background[markers == 0] = 1
        background[markers != 0] = 0


        self.__background = background

        self.__clean()

        self.__mask = cv2.morphologyEx(self.__mask.astype(np.uint8), cv2.MORPH_CLOSE,  conv,iterations=4).astype(np.float32)
        self.__cleaned = cv2.blur(self.__cleaned,(3,3))

        return Image(self.__cleaned.astype(np.float32)), Image(self.__mask.astype(np.float32)), Image(self.__mask.astype(np.float32))
