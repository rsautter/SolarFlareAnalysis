from libc.math cimport pow, sqrt, cos, sin
from math import radians

from libc.math cimport pow, sqrt, cos, sin
from math import radians

cimport cython 

cdef class Ellipse:
    cdef float rp, id
    cdef float angle, petrosianMag, sky, maxRad, minRad, fwhm
    cdef float posx, posy

    # Warning! - this builds an empty object
    cdef void setEmpty(self):
        print("Warning! - Empty Ellipse Object")
        self.rp = 0.0
        self.id = 0.0
        self.posx = 0.0
        self.posy = 0.0
        self.fwhm = 0.0
        self.petrosianMag = 0.0
        self.sky = 0.0
        self.angle = 0.0
        self.maxRad = 0.0
        self.minRad = 0.0

    def __cinit__(self, data=None, calibratedData=True):
        cdef float angulo,scaleA,scaleB
        self.rp = float(data['PETRO_RADIUS'])
        self.id = float(data['NUMBER'])
        self.posx = float(data['X_IMAGE'])
        self.posy = float(data['Y_IMAGE'])
        angulo = radians(float(data['THETA_IMAGE']))
        scaleA = float(data['A_IMAGE'])
        scaleB = float(data['B_IMAGE'])
        self.petrosianMag = float(data['FLUX_PETRO'])
        self.fwhm = float(data['FWHM_IMAGE'])
        self.sky = float(data['BACKGROUND'])
        if (calibratedData):
            self.petrosianMag = pow(10.0,-0.4*self.petrosianMag)
        self.angle = angulo
        self.maxRad = scaleA*self.rp
        self.minRad = scaleB*self.rp
 
    ##Encontra a escala da elipse que corta x, y
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef float findScale(self, float x,float  y):
        cdef float dx = x-float(self.posx)
        cdef float dy = y-float(self.posy)
        return sqrt(pow((dx*cos(self.angle)+dy*sin(self.angle))/(self.maxRad),2.0) + \
                pow((dx*sin(self.angle)-dy*cos(self.angle))/(self.minRad),2.0))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef float getCartesianDist(self, float x, float y):
        cdef float dx = x-float(self.posx)
        cdef float dy = y-float(self.posy)
        return (dx**2.0+dy**2.0)**0.5

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef tuple findPoint(self, float scale,float rho):
        x, y = scale*self.maxRad*cos(rho), scale*self.minRad*sin(rho)
        x2, y2 = self.posx+x*cos(self.angle)-y*sin(self.angle), self.posy+x*sin(self.angle)+y*cos(self.angle)
        return x2, y2
