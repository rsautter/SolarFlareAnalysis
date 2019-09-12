import numpy
from CyMorph2.Metrics.Metric import Metric

class Entropy(Metric):

    #@profile
    def __init__(self,bins=100):
        self.__bins = bins

    def getMasked(self, img, mask):
        mat = numpy.array(img.getMat())
        mask = mask.getMat()

        line = mat[numpy.where(numpy.array(mask)< 0.5)]
        freq = numpy.array([0.0 for i in range(self.__bins)], dtype=numpy.float32)
        temp, binagem = numpy.histogram(line,self.__bins)
        somatorio = 0.0
        for i in range(self.__bins):
            somatorio = somatorio + temp[i]
        for i in range(self.__bins):
            freq[i] = float(temp[i])/float(somatorio)
        somatorio = 0.0
        for i in range(self.__bins):
            if freq[i]>0.0:
                somatorio = somatorio - freq[i]*numpy.log10(freq[i])
        coef = somatorio/numpy.log10(self.__bins)
        return coef

    def getName(self):
        return "H"

