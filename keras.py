# only for testing, this file is not used for the final model

from __future__ import print_function

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
import numpy as np
import pickle

# Getting preprocessed datasets
with open('/work/nilsple/data/preprocessed.pkl', 'rb') as f:
    datasets, word_to_ix = pickle.load(f)

x, y = datasets
# Embedding
max_features = len(word_to_ix)+1
maxlen = len(x[0])
embedding_size = 128

x , y = np.array(x), np.array(y)
y = to_categorical(y,3)
x_train, y_train = x[:int(0.9*len(x))],y[:int(0.9*len(x))]
x_test, y_test = x[int(0.9*len(x)):],y[int(0.9*len(x)):]

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
epochs = 2

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size,return_sequences=True))
model.add(LSTM(lstm_output_size))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)