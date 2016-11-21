# for images
from PIL import Image
import numpy
import os
import time

# variables
negativeImageDirectory = '../../Hey-Waldo-master/16/notwaldo'
positiveImageDirectory = '../../Hey-Waldo-master/16/waldo'

# read in negative images
negativeImageFiles = os.listdir( negativeImageDirectory )

# do the same but for the positive images
positiveImageFiles = os.listdir( positiveImageDirectory )

# numFiles = len(negativeImageFiles) + len(positiveImageFiles)

positiveDataSet = numpy.ndarray((len(positiveImageFiles), 16*16*3 + 1))
negativeDataSet = numpy.ndarray((len(negativeImageFiles), 16*16*3 + 1))

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
			currentArray = numpy.append( currentArray , pixelArray )

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
			currentArray = numpy.append( currentArray , pixelArray )

	currentArray = numpy.append(currentArray, 1)
	positiveDataSet[k] = currentArray

print time.time() - start
