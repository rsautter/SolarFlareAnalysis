import numpy
from Metric import Metric
from libc.math cimport log10, fabs, sqrt
from libc.stdlib cimport rand, RAND_MAX

class Concentration(Metric):
    #@profile
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self._nSamples=50
        self.minCut = 0.09
        self.k = 1.5


    def __findRadiusLuminosity(self, dists, concentrations, totalLum, percentage):
        lenDists = len(dists)
        if(percentage < 0.0) or (percentage > 1.0):
            raise Exception("Percentage Invalid for Concentration! Got"+ str(percentage))
        if(totalLum == 0.0):
            raise Exception("Invalid Total Concentration! Got "+str(totalLum))
        if(percentage == 0.0):
            return 0.0

        accAnterior = concentrations[0]/totalLum
        for i in range(1,lenDists):
            acc = concentrations[i]/totalLum

        # interpolate the distance
            if(acc >= percentage):
                dy = acc - accAnterior
                dx = float(dists[i]) - float(dists[i-1])
                # if found a baseline, return the minimum baseline distance (also avoid overflow)
                if(fabs(dy) < 1.e-08):
                    return float(dists[i])
                return float(dists[i-1])+dx*(percentage-accAnterior)/dy
            accAnterior = acc
        return float(max(dists))

    # Returns the percentage of points at a distance dist
    def  __samplingMatrixPoint(self, epointx, epointy, pointx, pointy,dist,nsamples):
         px, py = float(pointx), float(pointy)
         if(self.__getCartesianDist(px+0.5,py+0.5, epointx, epointy)<=dist) and (self.__getCartesianDist(px-0.5,py+0.5, epointx, epointy)<=dist):
             if(self.__getCartesianDist(px+0.5,py-0.5, epointx, epointy)<=dist) and (self.__getCartesianDist(px-0.5,py-0.5, epointx, epointy)<=dist):
                     return 1.0
         if(self.__getCartesianDist(px+0.5,py+0.5, epointx, epointy)>dist) and (self.__getCartesianDist(px-0.5,py+0.5, epointx, epointy)>dist):
             if(self.__getCartesianDist(px+0.5,py-0.5, epointx, epointy)>dist) and (self.__getCartesianDist(px-0.5,py-0.5, epointx, epointy)>dist):
                     return 0.0
         ninside = 0
         for i in range(nsamples):
              rx = (float(rand())/float(RAND_MAX))-0.5
              ry = (float(rand())/float(RAND_MAX))-0.5
              if(self.__getCartesianDist(px+rx ,py+ry, epointx, epointy) < dist):
                   ninside = ninside + 1
         return float(ninside)/float(nsamples)

    #casrtesian distance
    def __getCartesianDist(self,x1,y1,x2,y2):
        return sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))


    def __getAccumulatedLuminosity(self, img, ibcg, minDist = 0):
        mat = img
        bcg = ibcg
        nSamples=self._nSamples
        localMinDist = minDist

        if(nSamples<=0):
            raise Exception("Invalid number of samples!(Accumulated Lum). Got"+str(nSamples))
        w, h = len(mat[0]), len(mat)
        epx, epy = w/2.0, h/2.0
        nDists = len(mat)/2 -localMinDist
        dists = numpy.array([0.0 for i in range(nDists)],dtype=numpy.float32)
        acc = numpy.array([0.0 for i in range(nDists)],dtype=numpy.float32)
        for i in range(localMinDist,len(mat)/2):
            dists[i-localMinDist] = float(i)
        for d in range(nDists):
            acc[d] = 0.0
            for wit in range(w):
                for hit in range(h):
                    percentage = self.__samplingMatrixPoint(epx, epy,wit,hit,dists[d],nSamples)
                    acc[d] = acc[d] + (mat[hit, wit]-bcg[hit, wit])*percentage
        return (dists, acc)

    '''
        Measures galaxy brightness, according to the sky median and variation
    '''
    def __getTotalLum(self, dists, conc):

        minD = self.minCut
        cutDist = 0
        k = self.k
        dConc = numpy.gradient(conc)
        # Finding distance that dconc < 1%
        for cutDist in range(2,len(dConc)):
            if(fabs(dConc[cutDist]/conc[cutDist])<minD):
                break

        if(len(dConc)-1 == cutDist):
            raise Exception("Not Convergent Concentration!")

        median = numpy.median(dConc[cutDist:len(dConc)])

        sigmaVector = numpy.zeros(len(dConc)-cutDist)
        for i in range(len(sigmaVector)):
            sigmaVector[i] = fabs(dConc[i+cutDist] - median)

        sigma = numpy.median(sigmaVector)
        avg = 0.0
        n = 0.0
        for i in range(1,len(dConc)):
            if(fabs(dConc[i])<k*sigma):
                n = n + 1.0
                avg += conc[i]
        if n > 5:
            return avg/n, cutDist
        else:
            raise Exception("Not Convergent Concentration!")

    # Notice that the second matrix is used as background, not a mask !!!!
    # Concentration should not be applied to masks
    def getMasked(self, img, bcg):
        dists, acc = self.__getAccumulatedLuminosity(img.getMat(),bcg.getMat())
        total, cutDist = self.__getTotalLum(dists,acc)
        rn = self.__findRadiusLuminosity(dists, acc, total, self.p1)
        rd = self.__findRadiusLuminosity(dists, acc, total, self.p2)
        if(rd == 0.0):
            raise Exception("Zero Division Error in concentration!")
        return log10(rn/rd)

    def getName(self):
        return "C_{}_{}".format(self.p1,self.p2)

    @staticmethod
    def buildConcentration(version=3):
        if version==1:
            return Concentration(0.8,0.2)
        if version==2:
            return Concentration(0.9,0.5)
        elif version==3:
            return Concentration(0.65,0.35)
        else:
            raise Exception("Not found C{}".format(version))


