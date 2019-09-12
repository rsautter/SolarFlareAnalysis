import numpy
from CyMorph2.Metrics.Metric import Metric
from CyMorph2.Image.Image import Image

class Analysis:

    def __init__(self):
        self.__listMetrics = []
        self.__listMaskedMetrics = []

    def add(self, m):
        self.__listMetrics.append(m)

    def addMasked(self, m):
        self.__listMaskedMetrics.append(m)

    def evaluate(self, img,  mask=None):
        output = {}
        for m in self.__listMetrics:
            output.update({m.getName(): m.get(img)})
        if not(mask is None):
            for m in self.__listMaskedMetrics:
                output.update({"M"+m.getName(): m.getMasked(img,mask)})
        return output
