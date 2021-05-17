from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from PIL import Image
import os
from tqdm import trange
import pickle

import keras
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

def digit(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

directry = 'data'
hw = 8
filenames = sorted(os.listdir(directry + '/0'))
load_num = 1000 #len(filenames)

features = [[] for _ in range(load_num)]
for i in trange(load_num):
    filename = filenames[i]
    img0 = load_img(directry + '/0/' + filename, target_size=(64, 64))
    img1 = load_img(directry + '/1/' + filename, target_size=(64, 64))
    features[i].append(img_to_array(img0))
    features[i].append(img_to_array(img1))
    '''
    for y in range(hw):
        for x in range(hw):
            features[i][0][y][x] = min(img0[y][x], 1)
            features[i][1][y][x] = min(img1[y][x], 1)
    '''
target = [0.0 for _ in range(load_num)]
with open(directry + '/score.txt', 'r') as f:
    for i in trange(load_num):
        target[i] = float(f.readline())

features = np.asarray(features)
target = np.asarray(target)

print('all data', features.shape, target.shape)

features_train, features_test, target_train, target_test = train_test_split(features, target,test_size=0.01, random_state=0)

print('train', features_train.shape, target_train.shape)
print('test ', features_test.shape, target_test.shape)


model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=features_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))       # クラスは2個
model.add(Activation('softmax'))

# コンパイル
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

# 実行。出力はなしで設定(verbose=0)。
history = model.fit(features_train, target_train, batch_size=5, epochs=200, validation_data = (features_test, target_test), verbose = 0)

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
            label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
            label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()


plot_history(history)


'''
print('learning start')
learning = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3)) #Lasso(alpha=0.1,random_state=0)
model = learning.fit(features_train,target_train)
pickle.dump(model, open(filename, 'wb'))
print('learning end')
model = pickle.load(open(filename, 'rb'))
'''

errors = []
exacts = []
for i in range(len(target_test)):
    prediction = model.predict([features_test[i]])[0][0]
    exact = target[i]
    err = abs(prediction - exact)
    print(err, prediction, exact)
    errors.append(err)
    exacts.append(abs(exact))

print(sum(errors) / len(errors))
print(sum(exacts) / len(exacts))