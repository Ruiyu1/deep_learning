#codin：utf-8
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
print(np.array(data))
print(label)



permutation = np.random.permutation(data.shape[0])
shuffled_dataset = data[permutation, :,:]
shuffled_labels = label[permutation]
encoded=to_categorical(shuffled_labels)
