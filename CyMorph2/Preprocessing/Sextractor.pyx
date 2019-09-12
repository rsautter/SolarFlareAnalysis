import ConfigParser
import os
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy.io import fits
import numpy
import pandas as pd
cimport cython
from FitsReader import write, read
from Ellipse import Ellipse

class Sextractor(object):

    def __init__(self,xtraID=0):
        self.__xtraID = xtraID
        self.__dados = numpy.array([])
        self.__dicionario = numpy.array([])


    def preprocessImage(self,img):
        write("temp"+str(self.__xtraID)+".fits", img)
        self.__runSextractor("temp"+str(self.__xtraID)+".fits")
        self.__lockObjectAtCenter(img)

    def __runSextractor(self,filePath,par=[],value=[]):
        configFile = ConfigParser.ConfigParser()
        configFile.read('cfg/paths.ini')
        sexPath = configFile.get("Path","Sextractor")
        cmd = sexPath+" "+filePath
        for i in range(len(par)):
            cmd = cmd+" -"+par[i]+" "+str(value[i])
        cmd = cmd+" -CATALOG_NAME c"+str(self.__xtraID)+".cat  -CHECKIMAGE_TYPE SEGMENTATION,BACKGROUND -CHECKIMAGE_NAME "+str(self.__xtraID)+"_seg.fits,"+str(self.__xtraID)+"_bcg.fits -VERBOSE_TYPE QUIET"
        process = os.popen(cmd)
        log = process.read()
        self.__readSextractorOutput("c"+str(self.__xtraID)+".cat")

    def __lockObjectAtCenter(self, img):
        minDist = -1.0
        minLine =[]
        px, py = numpy.array(img.getMat()).shape
        px, py = px/2, py/2
        for index,line in self.__dataframe.iterrows():
            x, y = float(line['X_IMAGE']), float(line['Y_IMAGE'])
            dist = (px-x)**2.0+(py-y)**2.0
            if minDist<0.0 or dist<minDist:
                minLine = line
                minDist = dist
        if minDist<0.0:
            raise Exception("No object detected by Sextractor !")
        return minLine

    #funciona apenas com arquivos fits!
    def __lockObject(self, ra, dec, infilename):
        #recuperando coordenadas globais
        header = fits.getheader(infilename, 0)
        data = fits.getdata(infilename, 0)
        ylen, xlen = data.shape[0], data.shape[1]
        print("Tamanho:", ylen,xlen)
        CRPIX1 = float(header['CRPIX1'])
        CRPIX2 = float(header['CRPIX2'])
        CRVAL1 = float(header['CRVAL1'])
        CRVAL2 = float(header['CRVAL2'])
        CD1_1 = float(header['CD1_1'])
        CD1_2 = float(header['CD1_2'])
        CD2_1 = float(header['CD2_1'])
        CD2_2 = float(header['CD2_2'])
        if header['CTYPE1'] == 'DEC--TAN':
            CD1_1 = float(header['CD2_1'])
            CD2_1 = float(header['CD1_1'])
            CD1_2 = float(header['CD2_2'])
            CD2_2 = float(header['CD1_2'])
            CRVAL1 = float(header['CRVAL2'])
            CRVAL2 = float(header['CRVAL1'])
        w = wcs.WCS(infilename, relax=True)
        w.wcs.crpix = [CRPIX1, CRPIX2]
        w.wcs.crval = [CRVAL1, CRVAL2]
        w.wcs.cd = [[CD1_1, CD2_1], [CD1_2, CD2_2]]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        sc = SkyCoord(ra=ra, dec=dec, unit='deg')
        px, py = skycoord_to_pixel(sc,w,0)
        minDist = -1.0
        minLine =[]

        # buscando o objeto mais proximo
        for index,line in self.__dataframe.iterrows():
            x, y = float(line['X_IMAGE']), float(line['Y_IMAGE'])
            dist = (px-x)**2.0+(py-y)**2.0
            if minDist<0.0 or dist<minDist:
                minLine = line
        if minDist<0.0:
            raise Exception("No object detected by Sextractor !")
        return minLine

    def __readSextractorOutput(self,filename):
        fileArx = open(filename, 'r')
        dicionario = []
        dados = [[]]
        for line in fileArx:
            splitList = line.split()
            if(splitList[0] == '#'):
                dicionario.append(splitList[2])
            else:
                dados.append(splitList)
        self.__dados = numpy.array([ [float(dados[i][j]) for j in range(len(dados[i])) ] for i in range(1,len(dados))])
        self.__dicionario = numpy.array(dicionario)
        self.__dataframe = pd.DataFrame(self.__dados, columns=self.__dicionario)

