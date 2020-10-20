from dataAnalysis.util import *
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, Flatten
from tensorflow.keras.preprocessing import sequence
from tcn import TCN, compiled_tcn

max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 200
batch_size = 32

(x_train, y_train), (x_test, y_test) = get_train_test_DS()

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
input_shape = x_train[0].shape
y_train = np.array(y_train)
y_test = np.array(y_test)
model = compiled_tcn(return_sequences=False,
                         num_feat=1,
                         num_classes=2,
                         nb_filters=10,
                         kernel_size=10,
                         dilations=[1, 2, 4, 8, 16, 32],
                         nb_stacks=6,
                         max_len=200,
                         use_skip_connections=False)

model.summary()

model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
x_train=x_train.reshape(990,200,1)
y_train=x_train.reshape(2,1)
print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1,
          validation_data=(x_test, y_test))
