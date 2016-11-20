import time
import os
from PIL import Image
import random
import numpy
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.datasets import mnist

print("Reading Data...")
from readDataColor import positiveDataSet, negativeDataSet
print("Finished Reading Data.")

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)

# shuffle & split dataSet
numpy.random.shuffle(positiveDataSet)
numpy.random.shuffle(negativeDataSet)

trainingData = numpy.append(positiveDataSet[:int(len(positiveDataSet) * 0.7)],
    negativeDataSet[:int(len(negativeDataSet) * 0.7)], axis=0)

testingData = numpy.append(positiveDataSet[int(len(positiveDataSet) * 0.7):],
    negativeDataSet[int(len(negativeDataSet) * 0.7):], axis=0)

# dataSet[:int(len(dataSet) * 0.9)]
# testingData = dataSet[int(len(dataSet) * 0.9):]

trainingDataX = trainingData[: , :len(trainingData[0]) - 1 ]
trainingDataY = trainingData[: , len(trainingData[0]) - 1 ]
trainingDataY = trainingDataY.reshape((-1,1))

testingDataX = testingData[: , :len(testingData[0]) - 1 ]
testingDataY = testingData[: , len(testingData[0]) - 1 ]
testingDataY = testingDataY.reshape((-1,1))

def init_model():
    start_time = time.time()
    print 'Compiling Model ... '
    model = Sequential()
    model.add(Dense(16*16*3/2, input_dim=16*16*3))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.4))

    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    rms = RMSprop()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    print 'Model compiled in {0} seconds'.format(time.time() - start_time)
    return model

def run_network(data=None, model=None, epochs=3, batch=256):
    try:
        start_time = time.time()
        if data is None:
            X_train, X_test, y_train, y_test = load_data()
        else:
            X_train, X_test, y_train, y_test = data

        if model is None:
            model = init_model()

        history = LossHistory()

        print 'Training model...'
        model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch,
                  callbacks=[history],
                  validation_data=(X_test, y_test), verbose=2)

        print "Training duration : {0}".format(time.time() - start_time)
        score = model.evaluate(X_test, y_test, batch_size=16)

        print "Network's test score [loss, accuracy]: {0}".format(score)




        inputDirectoryNames = [ '../../Hey-Waldo-master/64/waldo/' , '../../Hey-Waldo-master/64/notwaldo/' ]
        for dir in range(1):
            testFiles = os.listdir(inputDirectoryNames[dir])
            random.shuffle(testFiles)
            testFiles = testFiles [0:10]
            testFiles = [inputDirectoryNames[dir]+ a for a in testFiles]
        for t in testFiles:
            print t

        splice(testFiles[0] , model)
            #for t in testFiles:
                #print t
                #read in the file
                #while waldo is not dected AND theres still some left in the file
                    #get subarray
                    #feed subarray to model
                #if the model is positive:
                    # rates[dir * 2] += 1
                #else
                    #rates[dir*2 + 1] += 1




        #return model, history.losses
        return model
    except KeyboardInterrupt:
        print ' KeyboardInterrupt'
        return model, history.losses

def splice(fileName , model):
    img = Image.open(fileName)
    width, height = img.size
    for x in range(width - 16 + 1):
        for y in range(height - 16+ 1):
            #print x , y
            bbox = (x, y, x+ 16, y + 16)
            working_slice = img.crop(bbox)
            flattened = numpy.array(working_slice).flatten
            predictions = model.predict(self, flattened, batch, verbose=0)
            print size(predictions)
            print predictions
            #feed 32 x 32 x 3 
 

dataTuple = ( trainingDataX , testingDataX , trainingDataY , testingDataY )
annModel = init_model()
model , losses = run_network( dataTuple , annModel )
run_network( dataTuple , annModel )















