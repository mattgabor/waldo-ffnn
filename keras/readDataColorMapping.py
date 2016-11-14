# for images
from PIL 							import Image

import numpy
import os
import time
from math import sqrt
# variables
negativeImageDirectory = '../Hey-Waldo-master/64/notwaldo'
positiveImageDirectory = '../Hey-Waldo-master/64/waldo'

# negativeImageDirectory = './Hey-Waldo-master/64/notwaldosmall'
# positiveImageDirectory = './Hey-Waldo-master/64/waldosmall'

# read in negative images
negativeImageFiles = os.listdir( negativeImageDirectory )

# do the same but for the positive images
positiveImageFiles = os.listdir( positiveImageDirectory )

numFiles = len(negativeImageFiles) + len(positiveImageFiles)

# dataSet = numpy.ndarray((numFiles, 64*64 + 1))
positiveDataSet = numpy.ndarray((len(positiveImageFiles), 64*64 + 1))
negativeDataSet = numpy.ndarray((len(negativeImageFiles), 64*64 + 1))

imageMapping = {
	(0,0,0):0,
	(255,0,0):1,
	(0,255,0):2,
	(0,0,255):3,
	(255,255,0):4,
	(255,0,255):5,
	(0,255,255):6,
	(255,255,255):7
}

start = time.time()

for k in range(len(negativeImageFiles)):
	fi = negativeImageFiles[k]
	# open image
	currentArray = numpy.array( [] )
	fullPath 	 = negativeImageDirectory + '/' + fi
	currentImage = Image.open( fullPath )
	imagePixels  = currentImage.load()
	imageSize    = currentImage.size
	# read pixel values
	for i in range( imageSize[0] ):
		for j in range( imageSize[1] ):
			pixelArray = [ imagePixels[i,j][0] , imagePixels[i,j][1] , imagePixels[i,j][2] ]
			minPixelDistance = float('inf')
			minPixelMapping = -1
			for key in imageMapping:
				currentDistance = sqrt((pixelArray[0] - key[0])**2 + (pixelArray[1] - key[1])**2 + (pixelArray[2] - key[2])**2)
				if( currentDistance < minPixelDistance ):
					minPixelDistance = currentDistance
					minPixelMapping = imageMapping[key]

			currentArray = numpy.append( currentArray , minPixelMapping )

	# append to dataset
	currentArray = numpy.append(currentArray, 0)
	negativeDataSet[k] = currentArray


for k in range(len(positiveImageFiles)):
	fi = positiveImageFiles[k]
	currentArray = numpy.array( [] )
	fullPath     = positiveImageDirectory + '/' + fi
	currentImage = Image.open( fullPath )
	imagePixels  = currentImage.load()
	imageSize    = currentImage.size
	for i in range( imageSize[0] ):
		for j in range( imageSize[1] ):
			pixelArray = [ imagePixels[i,j][0] , imagePixels[i,j][1] , imagePixels[i,j][2] ]
			minPixelDistance = float('inf')
			minPixelMapping = -1
			for key in imageMapping:
				currentDistance = sqrt((pixelArray[0] - key[0])**2 + (pixelArray[1] - key[1])**2 + (pixelArray[2] - key[2])**2)
				if( currentDistance < minPixelDistance ):
					minPixelDistance = currentDistance
					minPixelMapping = imageMapping[key]
			currentArray = numpy.append( currentArray , minPixelMapping )

	currentArray = numpy.append(currentArray, 1)
	positiveDataSet[k] = currentArray

print time.time() - start
