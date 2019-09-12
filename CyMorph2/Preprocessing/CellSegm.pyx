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

    def preprocessImage(self, img):
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


        #Iniciando limpeza (removendo nucleo da celula)...
        isLessThan = np.where(cv2.blur(newMat,(15,15))<110)

        x, y = isLessThan
        rnewMat = np.zeros(newMat.shape)
        rnewMat[isLessThan] = 1
        ret, rnewMat = cv2.connectedComponents(rnewMat.astype(np.uint8))

        for blob in np.unique(rnewMat):
            if blob ==0:
                continue
            x, y = np.where(rnewMat == blob)
            rx, ry = np.where(rnewMat != blob)
            minX, maxX = np.min(x)-2,np.max(x)+2
            minY, maxY = np.min(y)-2,np.max(x)+2
            arg = np.where((rx>=minX) & (rx<=maxX))
            rx, ry = rx[arg], ry[arg]
            arg = np.where((ry>=minY) & (ry<=maxY))
            rx,ry = rx[arg], ry[arg]
            z = np.array(newMat[rx,ry])
            if len(z)>1:
                newMat[x, y] = np.random.normal(np.average(z),np.sqrt(np.std(z)),len(x))

        #Fazendo um threshold
        ret, thresh = cv2.threshold(newMat,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        ret, conn = cv2.connectedComponents(thresh.astype(np.uint8))
        maxBlob = 1
        lenMaxBlob = 0
        for blob in np.unique(conn):
            if blob ==0:
                continue
            x, y = np.where(conn == blob)
            if len(x)>= lenMaxBlob:
                maxBlob = blob
                lenMaxBlob = len(x)
        mask = conn.copy()
        mask[conn!=maxBlob] = 0
        mask[conn==maxBlob] = 1
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,  conv).astype(np.float32)
        return Image(mask)