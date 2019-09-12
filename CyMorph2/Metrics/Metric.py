import numpy
from CyMorph2.Image.Image import Image

class Metric:
    def get(self, img):
        mask = Image(numpy.zeros(img.getMat().shape).astype(numpy.float32()))
        return self.getMasked(img,mask)

    def getMasked(self, img, mask):
        print("Not implemented!")
        pass

    def getName(self):
        print("Not implemented!")
        pass
