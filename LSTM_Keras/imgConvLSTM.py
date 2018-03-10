#https://github.com/flyyufelix/VizDoom-Keras-RL/blob/4fc27ce3d400eba5422d39e2fad565d0503a6149/networks.py
import numpy as np
from keras.utils.vis_utils import plot_model  

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import TimeDistributed
from keras.layers import Input, Convolution2D
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.optimizers import Adam

model = Sequential()
# conv2d (batch_size, height, width, channels).
# And the TimeDistributed will require an additional dimension: (batch_size, frames, height, width, channels)

model.add(TimeDistributed(Conv2D(filters=1, kernel_size = [2, 2], subsample=[2,2], activation='relu'), input_shape=(20, 10,10,3)))
#model.add(TimeDistributed(Conv2D(filters=64, kernel_size = (4, 4), subsample=(2,2), activation='relu')))
#model.add(TimeDistributed(Conv2D(filters=64, kernel_size = (3, 3), activation='relu')))
model.add(TimeDistributed(Flatten()))

# Use all traces for training
#model.add(LSTM(512, return_sequences=True,  activation='tanh'))
#model.add(TimeDistributed(Dense(output_dim=action_size, activation='linear')))
action_size = 4
learning_rate = 0.01
# Use last trace for training
model.add(LSTM(512,  activation='tanh'))
model.add(Dense(output_dim=action_size, activation='linear'))

adam = Adam(lr=learning_rate)
model.compile(loss='mse',optimizer=adam)

plot_model(model, to_file='./modelPlot/LSTM_Conv_img.png',show_shapes=True)
