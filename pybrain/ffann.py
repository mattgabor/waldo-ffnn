# GOAL: Compute training error from all samples

# general dependencies
import numpy as np

import random

# pybrain dependencies
from pybrain.datasets            import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure           import FeedForwardNetwork, FullConnection
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork

# pylab dependencies for plotting
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
import matplotlib as mpl

import sys

print("Reading Data...")
from readData import dataSet
print("Finished Reading Data.")

# split data into testing and training data
testingDataTemp , trainingDataTemp = dataSet.splitWithProportion( 0.1 );

# reconvert to fix class issue
testingData = ClassificationDataSet( 64*64*3 , nb_classes = 2 );
for n in xrange( 0 , testingDataTemp.getLength() ):
	testingData.addSample( testingDataTemp.getSample( n )[ 0 ] , testingDataTemp.getSample( n )[ 1 ] );

trainingData = ClassificationDataSet( 64*64*3 , nb_classes = 2 );
for n in xrange( 0 , trainingDataTemp.getLength() ):
	trainingData.addSample( trainingDataTemp.getSample( n )[ 0 ] , trainingDataTemp.getSample( n )[ 1 ] );

# reencode outputs, necessary for training accurately
testingData._convertToOneOfMany();
trainingData._convertToOneOfMany();


##### BUILD ANN #####
# build feed-forward multi-layer perceptron ANN
fnn = FeedForwardNetwork()

# create layers: 9 input layer nodes (8 features + 1 bias), 3 hidden layer nodes, 10 output layer nodes
bias = BiasUnit(name='bias unit')
input_layer = LinearLayer(64*64*3, name='input layer')
hidden_layer = SigmoidLayer(64*64*3/2, name='hidden layer')
output_layer = SigmoidLayer(2, name='output layer')

# create connections with full connectivity between layers
bias_to_hidden = FullConnection(bias, hidden_layer, name='bias-hid')
bias_to_output = FullConnection(bias, output_layer, name='bias-out')
input_to_hidden = FullConnection(input_layer, hidden_layer, name='in-hid')
hidden_to_output = FullConnection(hidden_layer, output_layer, name='hid-out')

# add layers & connections to network
fnn.addModule(bias)
fnn.addInputModule(input_layer)
fnn.addModule(hidden_layer)
fnn.addOutputModule(output_layer)
fnn.addConnection(bias_to_hidden)
fnn.addConnection(input_to_hidden)
fnn.addConnection(hidden_to_output)
fnn.addConnection(bias_to_output)
fnn.sortModules()

# set up trainer that takes network & training dataset as input

# train model until convergence
trainer = BackpropTrainer(fnn, dataset=trainingData, verbose=True, learningrate = 1)

print("Starting Training...")
trainer.trainEpochs(1)
print("Finished Training.")

# results of training
allresult = percentError(trainer.testOnClassData(), dataSet['class'])
print ("\nepoch: %4d" % trainer.totalepochs, \
"  total error: %5.2f%%" % allresult)
