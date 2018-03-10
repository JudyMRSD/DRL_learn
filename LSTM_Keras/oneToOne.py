# Sequence Learning Problem
# https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
# example: echo program.
# input sequence: [0.0, 0.2, 0.4, 0.6, 0.8] 
# output sequence: [0.0, 0.2, 0.4, 0.6, 0.8] 


import numpy as np
from keras.utils.vis_utils import plot_model  

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# prepare sequence
length = 5
seq = np.array([i/float(length) for i in range(length)])
print("input seq", seq)
# input for LSTM must be 3D
# samples(len(seq)), time steps(1), features(1)
X = seq.reshape(len(seq), 1, 1)
y = seq.reshape(len(seq), 1)
# define LSTM configuration
n_neurons = length
n_batch = length
n_epoch = 1000
# create LSTM 
model = Sequential()
# The first hidden layer will be an LSTM with 5 units
model.add(LSTM(n_neurons, input_shape=(1,1)))
# The output layer is a fully-connected layer with 5 neurons.
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
plot_model(model, to_file='./modelPlot/OneToOne.png',show_shapes=True)

# train LSTM
model.fit(X, y, epochs = n_epoch, batch_size = n_batch, verbose=0)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result:
    print('%.1f' % value)



























