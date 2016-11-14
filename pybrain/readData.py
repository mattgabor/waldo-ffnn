# for images
from PIL 							import Image

# for network
from pybrain.datasets 				import ClassificationDataSet
from pybrain.utilities				import percentError
from pybrain.tools.shortcuts		import buildNetwork
from pybrain.supervised.trainers	import BackpropTrainer
from pybrain.structure.modules 		import SigmoidLayer
from pybrain.structure.modules   	import LinearLayer, BiasUnit
from pybrain.structure           	import FeedForwardNetwork, FullConnection

import numpy
import os

# variables
# negativeImageDirectory = './Hey-Waldo-master/64/notwaldo'
# positiveImageDirectory = './Hey-Waldo-master/64/waldo'

negativeImageDirectory = './Hey-Waldo-master/64/notwaldosmall'
positiveImageDirectory = './Hey-Waldo-master/64/waldosmall'

# read in negative images
dataSet = ClassificationDataSet( 64*64*3 , nb_classes=2 )
negativeImageFiles = os.listdir( negativeImageDirectory )

for fi in negativeImageFiles:
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
	dataSet.appendLinked( currentArray , 0 );

# do the same but for the positive images
positiveImageFiles = os.listdir( positiveImageDirectory )

for fi in positiveImageFiles:
	currentArray = numpy.array( [] )
	fullPath     = positiveImageDirectory + '/' + fi
	currentImage = Image.open( fullPath )
	imagePixels  = currentImage.load()
	imageSize    = currentImage.size
	for i in range( imageSize[0] ):
		for j in range( imageSize[1] ):
			pixelArray = [ imagePixels[i,j][0] , imagePixels[i,j][1] , imagePixels[i,j][2] ]
			currentArray = numpy.append( currentArray , pixelArray )
	dataSet.appendLinked( currentArray , 1 );
