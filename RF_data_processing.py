import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
from numpy.random import seed
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Dense, Merge
from keras.utils import to_categorical


filepath='/Users/ruiyu/Documents/RF_test_data'
pathDir =  os.listdir(filepath)
real_part = []
img_part = []
label = []
for i,allDir in enumerate(pathDir):
    if allDir[0]=='.':continue
    for file in os.listdir(filepath+'/'+allDir):
        if file[0] == '.': continue
        label.append(i)


        file = open(filepath+'/'+allDir+'/'+file)
        for c in file.readlines():
            c_array = c.split()
            real = float(c_array[0])
            img = float(c_array[1])


            real_part.append(real)
            img_part.append(img)

real_ = np.array(real_part)
real_ = np.reshape(real_,(32,576))
img_ = np.array(img_part)
img_ = np.reshape(img_,(32,576))
label_ = np.array(label)

label_ = np.reshape(label_,(32))
permutation = np.random.permutation(real_.shape[0])
print(permutation)
shuffled_real = real_[permutation,:]
shuffled_img = img_[permutation,:]
shuffled_labels = label_[permutation]

one_hot = to_categorical(shuffled_labels)
print(shuffled_real.shape)
print(shuffled_img.shape)
print(one_hot.shape)




branch1 = Sequential()
branch1.add(Dense(3600, input_shape=(576,), init='normal', activation='relu'))


branch2 = Sequential()
branch2.add(Dense(32, input_shape=(576,), init='normal', activation='relu'))

branch2.add(Dense(16, init='normal', activation='relu', W_constraint=maxnorm(5)))

branch2.add(Dense(4, init='normal', activation='relu', W_constraint=maxnorm(5)))


model = Sequential()
model.add(Merge([branch1, branch2], mode='concat'))
model.add(Dense(9, init='normal', activation='sigmoid'))
sgd = SGD(lr=0.1, momentum=0.9, decay=0, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
a=model.fit([real_, img_], one_hot, batch_size=20, nb_epoch=10, verbose=1)