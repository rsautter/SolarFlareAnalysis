import numpy
from math import radians, sqrt,pow,atan2,pi,fabs
from scipy.spatial import Delaunay as Delanuay

from CyMorph2.Metrics.Metric import Metric

class GPA(Metric):
    def __init__(self, gmoment=[2], mtol=0.01,atol=0.01):
        self.gmoment = gmoment
        self.mtol = mtol
        self.ftol = atol

    def __reset(self):
        # default value
        self.cx = -1
        self.cy = -1

        # percentual Ga proprieties
        self.cols = len(self.mat[0])
        self.rows = len(self.mat)
        self.totalVet = self.rows * self.cols
        self.totalAssimetric = self.rows * self.cols
        self.removedP = numpy.array([],dtype=numpy.int32)
        self.nremovedP = numpy.array([],dtype=numpy.int32)
        self.phaseDiversity = 0.0

    def _setGradients(self,img):

        dx, dy = img.gradient()
        gy, gx = dy.getMat(), dx.getMat()
        w, h = len(gx[0]),len(gx)
        
       
        self.maxGrad = -1.0
        for i in range(w):
            for j in range(h):
                if(self.maxGrad<0.0) or (sqrt(pow(gy[j, i],2.0)+pow(gx[j, i],2.0))>self.maxGrad):
                    self.maxGrad = sqrt(pow(gy[j, i],2.0)+pow(gx[j, i],2.0))
        
        #initialization
        self.gradient_dx=numpy.array([[gx[j, i] for i in range(w) ] for j in range(h)])
        self.gradient_dy=numpy.array([[gy[j, i] for i in range(w) ] for j in range(h)])

        # copying gradient field to asymmetric gradient field
        self.gradient_asymmetric_dx = numpy.array([[gx[j, i] for i in range(w) ] for j in range(h)])
        self.gradient_asymmetric_dy = numpy.array([[gy[j, i] for i in range(w) ] for j in range(h)])

        # calculating the phase and mod of each vector
        self.phases = numpy.array([[atan2(gy[j, i],gx[j, i]) if atan2(gy[j, i],gx[j, i]) > 0.0 else 2.0*pi+atan2(gy[j, i],gx[j, i]) 
                                     for i in range(w) ] for j in range(h)])
        self.mods = numpy.array([[sqrt(pow(gy[j, i],2.0)+pow(gx[j, i],2.0)) for i in range(w) ] for j in range(h)])


    def _angleDifference(self,a1,a2):
        return min(fabs(a1-a2), fabs(fabs(a1-a2)-2.0*pi))


    def _update_asymmetric_mat(self,index_dist,dists):
        mods = self.mods
        phases = self.phases
        maxGrad = self.maxGrad
        rows,cols = self.rows,self.cols

        mtol, ftol, ptol = self.mtol, self.ftol, 2.0

        # distances loop
        for ind in range(0, len(index_dist)):

            x2, y2 =[], []
            for py in range(rows):
                for px in range(cols):
                    if (fabs(dists[py, px]-index_dist[ind]) <= fabs(ptol)):
                        x2.append(px)
                        y2.append(py)
            x, y =numpy.array(x2,dtype=numpy.int32), numpy.array(y2,dtype=numpy.int32)
            lx = len(x)

            # compare each point in the same distance
            for i in range(lx):
                px, py = x[i], y[i]
                if (mods[py, px]/maxGrad <= mtol):
                    self.gradient_asymmetric_dx[py, px] = 0.0
                    self.gradient_asymmetric_dy[py, px] = 0.0
                if(px<2 or px>rows-2 or py<2 or py>cols-2):
                    self.gradient_asymmetric_dx[py, px] = 0.0
                    self.gradient_asymmetric_dy[py, px] = 0.0
                    continue
                for j in range(lx):
                    px2, py2 = x[j], y[j]
                    if (fabs((mods[py, px]- mods[py2, px2])/maxGrad )<= mtol):
                        if (fabs(self._angleDifference(phases[py, px], phases[py2, px2])-pi)  <= ftol) :
                            self.gradient_asymmetric_dx[py, px] = 0.0
                            self.gradient_asymmetric_dy[py, px] = 0.0
                            self.gradient_asymmetric_dx[py2, px2] = 0.0
                            self.gradient_asymmetric_dy[py2, px2] = 0.0
                            break

        self.totalVet = 0
        self.totalAssimetric = 0
        nremovedP = []
        removedP = []

        for j in range(self.rows):
            for i in range(self.cols):
                if (sqrt(pow(self.gradient_asymmetric_dy[j,i],2.0)+pow(self.gradient_asymmetric_dx[j,i],2.0)) <= mtol):
                    removedP.append([j,i])
                    self.totalVet = self.totalVet+1
                else:
                    nremovedP.append([j,i])
                    self.totalVet = self.totalVet+1
                    self.totalAssimetric = self.totalAssimetric+1
        self.totalVet = rows*cols-2*rows-2*cols+4
        if(len(nremovedP)>0):
            self.nremovedP = numpy.array(nremovedP,dtype=numpy.int32)
        if(len(removedP)>0):
            self.removedP = numpy.array(removedP,dtype=numpy.int32)

    def _G2V(self):
        somax = 0.0
        somay = 0.0
        smod = 0.0
        hasMask = hasattr(self,'mask')
        if hasMask:
            mask = self.mask.getMat()
        if(self.totalAssimetric<1):
            return 0.0
        for i in range(self.totalAssimetric):
            if hasMask:
                if mask[self.nremovedP[i,0],self.nremovedP[i,1]] < 0.5:
                    continue
            phase = self.phases[self.nremovedP[i,0],self.nremovedP[i,1]]
            mod = self.mods[self.nremovedP[i,0],self.nremovedP[i,1]]
            somax += self.gradient_dx[self.nremovedP[i,0],self.nremovedP[i,1]]
            somay += self.gradient_dy[self.nremovedP[i,0],self.nremovedP[i,1]]
            smod += mod
        if smod <= 0.0:
            return 0.0
        alinhamento = sqrt(pow(somax,2.0)+pow(somay,2.0))/smod
        return alinhamento

    def _G3V(self):
        sumPhases = 0.0
        for i in range(self.totalAssimetric-1):
            x1,y1 = self.nremovedP[i,0],self.nremovedP[i,1]
            for j in range(i+1,self.totalAssimetric):
                x2,y2 = self.nremovedP[j,0],self.nremovedP[j,1]
                sumPhases += self.distAngle(self.phases[x1,y1],self.phases[x2,y2])
        div = ((self.totalAssimetric)*(self.totalAssimetric-1))/2
        variety = sumPhases / float(div)
        return variety


    def _G4(self):
        w, h = self.cols,self.rows
        if(len(self.nremovedP[:,0])>3):
            self.totalAssimetric = len(self.nremovedP[:,0])
        else:
            self.totalAssimetric = 0
        nmods = self.mods
        sumMod = 0.0
        for i in self.nremovedP:
            sumMod += self.mods[i[0],i[1]]

        for m in range(len(nmods)):
            for n in range(len(nmods[m])):
                nmods[m,n] = nmods[m,n]/sumMod

        self.cvet = numpy.array([-nmods[i[0],i[1]]*numpy.log(nmods[i[0],i[1]])+1.0j*nmods[i[0],i[1]]*self.phases[i[0],i[1]]
                                 if nmods[i[0],i[1]] > 0.0 else
                                 1.0j*nmods[i[0],i[1]]*self.phases[i[0],i[1]]
                    for i in self.nremovedP],dtype=numpy.complex64)
        self.G4 = numpy.sum(self.cvet)
        return self.G4

    def _G3(self):
        if(len(self.nremovedP[:,0])>3):
            self.totalAssimetric = len(self.nremovedP[:,0])
        else:
            self.totalAssimetric = 0
        self.phaseDiversity = self._G3V()
        self.G3 = ((self.totalAssimetric)/float(self.totalVet))*(2.0-self.phaseDiversity)
        return self.G3

    def _G2(self):
        if len(self.nremovedP) == 0:
            self.modDiversity = 0.0
            self.G2 = ((self.totalAssimetric)/float(self.totalVet))*(2.0-self.modDiversity)
            return self.G2
        if len(self.nremovedP[:,0])>3 :
            if hasattr(self,'mask'):
                self.totalAssimetric = 0
                mask = self.mask.getMat()
                self.totalVet = len(numpy.where(numpy.array(mask)>0.5)[0])
                for i in range(len(self.nremovedP)):
                    if mask[self.nremovedP[i,0],self.nremovedP[i,1]]>0.5:
                        self.totalAssimetric += 1
            else:
                self.totalAssimetric = len(self.nremovedP[:,0])
        else:
            self.totalAssimetric = 0
        self.modDiversity = self._G2V()
        self.G2 = ((self.totalAssimetric)/float(self.totalVet))*(2.0-self.modDiversity)
        return self.G2

    def _G1(self,tol):

        triangulation_points = []
        hasMask = hasattr(self,'mask')
        if hasMask:
            mask = self.mask.getMat()
        for i in range(self.rows):
            for j in range(self.cols):
                mod = (self.gradient_asymmetric_dx[i, j]**2+self.gradient_asymmetric_dy[i, j]**2)**0.5
                if hasMask:
                    if mask[i,j] < 0.5:
                        continue
                if mod > tol:
                    triangulation_points.append([j+0.5*self.gradient_asymmetric_dx[i, j], i+0.5*self.gradient_asymmetric_dy[i, j]])
        triangulation_points = numpy.array(triangulation_points)
        self.n_points = len(triangulation_points)
        if self.n_points < 3:
            self.n_edges = 0
            self.G1 = 0
        else:
            triangles = Delanuay(triangulation_points)
            neigh = triangles.vertex_neighbor_vertices
            self.n_edges = len(neigh[1])/2
            self.G1 = float(self.n_edges-self.n_points)/float(self.n_points)
        return self.G1

    def getMasked(self, img, mask):

        self.mask = mask
        out = self.get(img)
        delattr(self,'mask')
        return out

    # This function estimates both asymmetric gradient coeficient (geometric and algebric), with the given tolerances
    def get(self,img):

        self.mat = img.getMat()
        self.__reset()
        self.cols = len(self.mat[0])
        self.rows = len(self.mat)

        self._setGradients(img)

        self.cy = float(self.cols)/2.0
        self.cx = float(self.rows)/2.0

        dists = numpy.array([[int(sqrt(pow(x-self.cx, 2.0)+pow(y-self.cy, 2.0))) \
                                                  for x in range(self.cols)] for y in range(self.rows)])
        uniq = numpy.unique(dists)

        # removes the symmetry in gradient_asymmetric_dx and gradient_asymmetric_dy:
        self._update_asymmetric_mat(uniq.astype(dtype=numpy.int32), dists.astype(dtype=numpy.int32))

        retGa = 0.0
        if self.gmoment == 2:
            res = self._G2()
        elif self.gmoment == 3:
            res = self._G3()
        elif self.gmoment == 4:
            res = self._G4()
        else:
            res = self._G1(self.mtol)

        return res


    def _wasRemoved(self, j,i):
        for rp in range(self.totalVet - self.totalAssimetric):
            if(self.removedP[rp,0] == j) and(self.removedP[rp,1] == i):
                return True
        return False

    def getName(self):
        return "G{}".format(self.gmoment)


