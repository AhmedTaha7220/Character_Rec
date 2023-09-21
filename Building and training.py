###### This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import librosa
import librosa.display
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
for filename in filenames:
print(os.path.join(dirname, filename))
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
reco=librosa.load('/kaggle/input/speech/data/one/00176480_nohash_0.wav')
import os
import librosa
import numpy as np
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping ,ModelCheckpoint
from keras import backend as k
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
train_audio_path = '/kaggle/input/commands'
labels = os.listdir(train_audio_path)
all_wave = []
all_label = []
for label in labels:
print(label)
waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
for wav in waves:
samples , sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav , sr= 8000)
samples = librosa.resample(samples , orig_sr=sample_rate, res_type='polyphase', target_sr=8000)
if (len(samples) == 8000):
all_wave.append(samples)
all_label.append(label)
le = LabelEncoder()
y = le.fit_transform(all_label)
classes = list(le.classes_)
from keras.utils import np_utils
y = np_utils.to_categorical(y , num_classes= len(labels))
all_wave = np.array(all_wave).reshape(-1 ,8000 ,1)
from sklearn.model_selection import train_test_split
x_tr ,x_val ,y_tr ,y_val = train_test_split(np.array(all_wave) ,np.array(y) ,stratify = y ,
test_size = 0.2 ,random_state = 777 ,shuffle =True)
k.clear_session()
inputs = Input(shape=(8000,1))
# first layer
conv = Conv1D(8 , 13, padding= 'valid' , activation ='relu' ,strides =1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)
# second layer
conv = Conv1D(16 , 11, padding= 'valid' , activation ='relu' ,strides =1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)
# third layer
conv = Conv1D(32 , 9, padding= 'valid' , activation ='relu' ,strides =1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)
# fourth layer
conv = Conv1D(64 , 7, padding= 'valid' , activation ='relu' ,strides =1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)
# Flatten
conv = Flatten()(conv)
# Dense 1
conv = Dense(256 , activation = 'relu')(conv)
conv = Dropout(0.3)(conv)
# Dense 2
conv = Dense(128 , activation = 'relu')(conv)
conv = Dropout(0.3)(conv)
outputs = Dense(len(labels) ,activation = 'softmax')(conv)
model = Model(inputs,outputs)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor= 'val_loss', mode= 'min', verbose= 1, patience= 10, min_delta= 0.0001)
mc = ModelCheckpoint('best_mode12_hdf5',monitor= 'val_loss',verbose= 1 ,save_best_only= True ,mode= 'max')
history = model.fit(x_tr ,y_tr , epochs=100 ,callbacks= [es ,mc] , batch_size=32 ,validation_data=(x_val,y_val))
pyplot.plot(history.history['loss'], label = 'train')
pyplot.plot(history.history['val_loss'], label = 'test')
pyplot.legend()
pyplot.show()
m=model.save("spec.model")
n=model.save("command.model")