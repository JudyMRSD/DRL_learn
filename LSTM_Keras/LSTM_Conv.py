# tutorial:
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
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



# Part 1.  load an preprocess data


# We need to load the IMDB dataset. 
# We are constraining the dataset to the top 5,000 words.
# We also split the dataset into train (50%) and test (50%) sets.
top_words = 5000
# Top most frequent words to consider. 
# Any less frequent word will appear as oov_char value in the sequence data.
(X_train, y_train),(X_test, y_test) = imdb.load_data(num_words=top_words)
print("X_train[0] shape", np.shape(X_train[0])) # (218,)
print("y_train[0] shape", np.shape(y_train[0]))
print("X_train[0]",X_train[0])
print("y_train[0]",y_train[0])# 1
# truncate and pad the input sequences so that they are all the same length for modeling
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen = max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen = max_review_length)
print("X_train[0] shape", np.shape(X_train[0])) # (500,)
print("y_train[0] shape", np.shape(y_train[0]))
print("X_train[0]",X_train[0])
print("y_train[0]",y_train[0])# 1

# Part 2.  Define, compile and fit LSTM model
embedding_vector_length = 32
model = Sequential()

# The first layer is the Embedded layer that uses 
# 32 (embedding_vector_length)length vectors to represent each word

# Turns positive integers (indexes) into dense vectors of fixed size. 
# eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
model.add(Embedding(top_words, embedding_vector_length, input_length = max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
# Because it is a binary classification problem, 
# log loss is used as the loss function (binary_crossentropy in Keras)

# The efficient ADAM optimization algorithm is used. 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# The model is fit for only 2 epochs because it quickly overfits the problem. 
# A large batch size of 64 reviews is used to space out weight updates.
print(model.summary())
plot_model(model, to_file='./modelPlot/LSTM_Conv.png',show_shapes=True)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)

# Part 3. Evaluate performance
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy:%.2f%%" %(scores[1]*100))










