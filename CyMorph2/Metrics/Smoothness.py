import numpy
import scipy.stats as stats
from CyMorph2.Image.Image import Image
from CyMorph2.Metrics.Metric import Metric

class Smoothness(Metric):

    #@profile
    def __init__(self, correlation, sDegree,version):
        self.__correlation = correlation
        self.__sDegree = sDegree
        self.__version = version

    def getMasked(self, img, mask):
        mat = img.getMat()
        w, h = len(mat[0]), len(mat)

        sm = img.filterButterworth2D(self.__sDegree)

        matSmooth = sm.getMat()
        localMask = mask.getMat()
        localMat = img.getMat()

        countNotMasked = 0
        self.v1 = []
        self.v2 = []
        for i in range(w):
           for j in range(h):
               if (localMask[j,i] < 0.5):
                   countNotMasked = countNotMasked + 1
                   self.v1.append(localMat[j,i])
                   self.v2.append(matSmooth[j,i])

        coef = 1.0-self.__correlation(self.v1, self.v2)[0]

        return coef

    @staticmethod
    def buildSmoothness(version=2,sDegree=0.5):
        if version==2:
            return Smoothness(stats.pearsonr, sDegree,version)
        elif version==3:
            return Smoothness(stats.spearmanr, sDegree,version)
        else:
            raise Exception("Not found correlation coefficient for S{}".format(version))

    def getName(self):
        return "S{}".format(self.__version)
