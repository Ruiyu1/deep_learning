from __future__ import print_function


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


import os
import numpy as np
from keras.utils import to_categorical


filepath='/Users/ruiyu/Documents/nonoise'
pathDir =  os.listdir(filepath)
data = []
label = []
for i,allDir in enumerate(pathDir):
    if allDir[0]=='.':continue
    for file in os.listdir(filepath+'/'+allDir):
        if file[0] == '.': continue
        data.append([np.loadtxt(filepath+'/'+allDir+'/'+file)])
        label.append(i)

data=np.array(data)
label=np.array(label)
data = np.reshape(data,(60,3600))



permutation = np.random.permutation(data.shape[0])
shuffled_dataset = data[permutation,:]
shuffled_labels = label[permutation]
encoded=to_categorical(shuffled_labels)


batch_size = 128
num_classes = 7
epochs = 20

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=3600))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(shuffled_dataset,encoded,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)
