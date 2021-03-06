# tutorial:
# Many to 1 :   
# LSTM input: batch size  x  time steps  x  num input features(32) 
#             time steps: number of words in a sentence = 50  
#             each word is represented by a vector of length 32
# LSTM output: batch size  x   num output featuers (1)
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# test result after 25000 epochs : Accuracy:85.85%

# 25000/25000 [==============================] - 608s 24ms/step - loss: 0.4329 - acc: 0.7978 - val_loss: 0.3687 - val_acc: 0.8427
# Epoch 2/3
# 25000/25000 [==============================] - 615s 25ms/step - loss: 0.2888 - acc: 0.8853 - val_loss: 0.3188 - val_acc: 0.8711
# Epoch 3/3
# 25000/25000 [==============================] - 580s 23ms/step - loss: 0.2504 - acc: 0.9008 - val_loss: 0.3332 - val_acc: 0.8585
# Accuracy:85.85%

import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils.vis_utils import plot_model  
import tensorflow as tf 
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
np.random.seed(7)


sess = tf.Session()


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
# 32 (embedding_vector_length)length vectors 
# to represent each word

# Turns positive integers (indexes) into dense vectors of fixed size. 
# eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
model.add(Embedding(top_words, embedding_vector_length, input_length = max_review_length))
# The next layer is the LSTM layer with 100 memory units 
model.add(LSTM(100))

# Finally, because this is a classification problem we use 
# a Dense output layer with a single neuron 
# and a sigmoid activation function 
# to make 0 or 1 predictions for the two classes (good and bad) in the problem.
model.add(Dense(1, activation='sigmoid'))

# Because it is a binary classification problem, 
# log loss is used as the loss function (binary_crossentropy in Keras)

# The efficient ADAM optimization algorithm is used. 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_writer = tf.summary.FileWriter('./log/simpleLSTM', sess.graph)

# The model is fit for only 2 epochs because it quickly overfits the problem. 
# A large batch size of 64 reviews is used to space out weight updates.
print(model.summary())
plot_model(model, to_file='./modelPlot/SimpleLSTM.png',show_shapes=True)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)

# Part 3. Evaluate performance
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy:%.2f%%" %(scores[1]*100))

train_writer.close()









