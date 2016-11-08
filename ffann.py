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

# write errors to file
sys.stdout = open('output/p2.txt', 'w+')

##### IMPORT DATA #####

# import data
X = np.genfromtxt('data/yeast.data', usecols=(1,2,3,4,5,6,7,8))
classes = np.genfromtxt('data/yeast.data', dtype=str, usecols=(9)).tolist()

# assign class values to integers for usability (Alphabetically)
classNames = ['CYT', 'ERL', 'EXC', 'ME1', 'ME2', 'ME3', 'MIT', 'NUC', 'POX', 'VAC']
classDict = {'CYT': 0, 'ERL': 1, 'EXC': 2, 'ME1': 3, 'ME2': 4, 'ME3': 5, 'MIT': 6, 'NUC': 7, 'POX': 8, 'VAC': 9}
Y = []

for name in classes:
    Y.append(classDict[name])

##### CONSTRUCT DATASET #####

# create ClassificationDataSet and add samples
ds = ClassificationDataSet(8, 1, nb_classes=10, class_labels=classNames)
for i in range(X.shape[0]):
    X_i = X[i, :].tolist()
    ds.addSample(X_i, Y[i])

ds._convertToOneOfMany()

##### BUILD ANN #####
# build feed-forward multi-layer perceptron ANN
fnn = FeedForwardNetwork()

# create layers: 9 input layer nodes (8 features + 1 bias), 3 hidden layer nodes, 10 output layer nodes
bias = BiasUnit(name='bias unit')
input_layer = LinearLayer(8, name='input layer')
hidden_layer = SigmoidLayer(3, name='hidden layer')
output_layer = SigmoidLayer(10, name='output layer')

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
trainer = BackpropTrainer(fnn, dataset=ds, verbose=False, learningrate = 1)
trainer.trainEpochs(100)

# results of training
allresult = percentError(trainer.testOnClassData(), ds['class'])
print ("\nepoch: %4d" % trainer.totalepochs, \
"  total error: %5.2f%%" % allresult)
