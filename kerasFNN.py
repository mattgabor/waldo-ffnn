import time
import numpy
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.datasets import mnist

print("Reading Data...")
from readDataKeras import dataSet
print("Finished Reading Data.")

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)

# shuffle & split dataSet

numpy.random.shuffle(dataSet)
trainingData = dataSet[:int(len(dataSet) * 0.9)]
testingData = dataSet[int(len(dataSet) * 0.9):]

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
    model.add(Dense(64*64*3/2, input_dim=64*64*3))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.4))

    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    rms = RMSprop()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    print 'Model compiled in {0} seconds'.format(time.time() - start_time)
    return model

def run_network(data=None, model=None, epochs=20, batch=256):
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
        return model, history.losses
    except KeyboardInterrupt:
        print ' KeyboardInterrupt'
        return model, history.losses

dataTuple = ( trainingDataX , testingDataX , trainingDataY , testingDataY )
annModel = init_model()
model , losses = run_network( dataTuple , annModel )
