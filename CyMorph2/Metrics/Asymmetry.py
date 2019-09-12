import numpy
import scipy.stats as stats
from CyMorph2.Image.Image import Image
from CyMorph2.Metrics.Metric import Metric

class Asymmetry(Metric):

    #@profile
    def __init__(self, correlation,version):
        self.__correlation = correlation
        self.__version = version

    def getMasked(self, img, mask):
        mat = img.getMat()
        w, h = len(mat[0]), len(mat)

        matInv = img.rotateImage(90.0)
        maskInv = mask.rotateImage(90.0)

        maskInverse = maskInv.getMat()
        matInverse = matInv.getMat()
        localMask = mask.getMat()
        localMat = img.getMat()

        countNotMasked = 0
        self.v1 = []
        self.v2 = []
        for i in range(w):
           for j in range(h):
               if (localMask[j,i] < 0.5) and (maskInverse[j,i] < 0.5):
                   countNotMasked = countNotMasked + 1
                   self.v1.append(localMat[j,i])
                   self.v2.append(matInverse[j,i])

        mv1 = numpy.max(self.v1)
        mv2 = numpy.max(self.v2)
        for i in range(countNotMasked):
            self.v1[i] = self.v1[i]/mv1
            self.v2[i] = self.v2[i]/mv2

        coef = 1.0-self.__correlation(self.v1, self.v2)[0]

        return coef

    def getName(self):
        return "A{}".format(self.__version)

    @staticmethod
    def buildAsymmetry(version=2):
        if version==2:
            return Asymmetry(stats.pearsonr,version)
        elif version==3:
            return Asymmetry(stats.spearmanr,version)
        else:
            raise Exception("Not found correlation coefficient for A{}".format(version))


