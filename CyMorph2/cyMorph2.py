import JpegReader
from GPA import GPA
from Asymmetry import Asymmetry
from Smoothness import Smoothness
from Entropy import Entropy
from Metric import Metric
from Analysis import Analysis
from Sextractor import Sextractor
from CellSegm import CellSegm
from Concentration import Concentration
import pandas as pd
import sys
import Image
import JpegReaderC
from ColoredCellSegm import ColoredCellSegm
import matplotlib.pyplot as plt

def cyMorph(img):
	a = Analysis()
	a.add(GPA(1, mtol=0.03, atol=0.03))
	a.add(GPA(2, mtol=0.03, atol=0.03))
	a.add(Asymmetry.buildAsymmetry(2))
	a.add(Asymmetry.buildAsymmetry(3))
	a.add(Smoothness.buildSmoothness(2))
	a.add(Smoothness.buildSmoothness(3))
	a.add(Entropy())
	return a.evaluate(img,mask)    

def main():
	a = Analysis()
	res = []
	# full image metrics
	'''
	a.add(GPA(1, mtol=0.03, atol=0.03))
	a.add(GPA(2, mtol=0.03, atol=0.03))

	a.add(Asymmetry.buildAsymmetry(2))
	a.add(Asymmetry.buildAsymmetry(3))
	a.add(Smoothness.buildSmoothness(2))
	a.add(Smoothness.buildSmoothness(3))
	a.add(Entropy())
	'''

	#a.add(Concentration.buildConcentration(3))

	# masked image metrics
	a.addMasked(GPA(1, mtol=0.03, atol=0.03))
	a.addMasked(GPA(2, mtol=0.03, atol=0.03))
	a.addMasked(Asymmetry.buildAsymmetry(2))
	a.addMasked(Asymmetry.buildAsymmetry(3))
	a.addMasked(Smoothness.buildSmoothness(2, sDegree=0.15))
	a.addMasked(Smoothness.buildSmoothness(3, sDegree=0.15))
	a.addMasked(Entropy(bins=200))

	input = open(sys.argv[1], "r")
	segmentador = CellSegm()
	for fileName in input:
		filePath = fileName.replace('\n','')
		img = JpegReader.read(filePath)
		mask = segmentador.preprocessImage(img)
		outFile = filePath.split('/')[len(filePath.split('/'))-1]

		JpegReader.write("masks/m" + outFile, mask)
		dict = a.evaluate(img,mask)
		dict.update({"File":outFile})
		res.append(dict)
		print(pd.DataFrame(res))
		print("\n")

	df = pd.DataFrame(res)
	with open("Result.csv", "w") as output:
		df.to_csv(output, index=False)
	print(df)


if __name__=="__main__":
	main()
