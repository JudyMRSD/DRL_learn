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
from keras.models import Model

# https://programtalk.com/python-examples/keras.layers.TimeDistributed/
action_size = 4
learning_rate = 0.01


input_layer = Input(shape = (20,10,10,3))

x = TimeDistributed(Conv2D(filters=1, kernel_size=[2, 2], activation="relu", strides=[2, 2]))(input_layer)

x = TimeDistributed(Flatten())(x)
x = LSTM(512,  activation='tanh')(x)
output_layer = Dense(activation="linear", units=4)(x)

model = Model(input_layer, output_layer)
adam = Adam(lr=learning_rate)
model.compile(loss='mse',optimizer=adam)

plot_model(model, to_file='./modelPlot/LSTM_Conv_img_2.png',show_shapes=True)
