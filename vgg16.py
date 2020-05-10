#All Necessary Imports
import numpy as np
import os
import time
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import model_from_json

#loading base model
base_model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
#freeze_layers(base_model)
base_model.summary()

# Freeze the layers except the last 4 layers
for layer in base_model.layers[:-3]:
    layer.trainable = False
# Check the trainable status of the individual layers
for layer in base_model.layers:
    print(layer, layer.trainable)
base_model.summary()

#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 5
# number of epochs to train
nb_epoch = 10

from keras import models
from keras import layers
from keras import optimizers
 
# Create the model
model = models.Sequential()
 
# Add the vgg convolutional base model
model.add(base_model)
 
# Add new layers
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(nb_classes, activation='softmax', name ='output'))
 
# Show a summary of the model. Check the number of trainable parameters
model.summary()

import pandas as pd
trainLabels = pd.read_csv("trainLabels.csv")
trainLabels.head()

import os

listing = os.listdir("images") 
np.size(listing)

from PIL import Image

# input image dimensions
img_rows, img_cols = 224, 224

immatrix = []
imlabel = []

for file in listing:
    base = os.path.basename("images/" + file)
    fileName = os.path.splitext(base)[0]
    imlabel.append(trainLabels.loc[trainLabels.image==fileName, 'level'].values[0])
    im = Image.open("images/" + file)   
    img = im.resize((img_rows,img_cols))
    rgb = img.convert('RGB')
    immatrix.append(np.array(rgb).flatten())

from sklearn.utils import shuffle

#converting images & labels to numpy arrays
immatrix = np.asarray(immatrix)
imlabel = np.asarray(imlabel)


data,Label = shuffle(immatrix,imlabel, random_state=2)
train_data = [data,Label]
type(train_data)

import matplotlib.pyplot as plt
import matplotlib
for i in range (20):
    img=immatrix[i].reshape(img_rows,img_cols,3)
    print('severity',imlabel[i])
    if(imlabel[i]>0):
        plt.imshow(img)

(X, y) = (train_data[0],train_data[1])
from sklearn.model_selection import train_test_split

# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

X_train = X_train.reshape(X_train.shape[0], img_cols, img_rows, 3)
X_test = X_test.reshape(X_test.shape[0], img_cols, img_rows, 3)


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

from keras.utils import np_utils

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

from keras.preprocessing.image import ImageDataGenerator

# create generators  - training data will be augmented images
validationdatagenerator = ImageDataGenerator()
traindatagenerator = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,rotation_range=15,zoom_range=0.1 )

batchsize=8
train_generator=traindatagenerator.flow(X_train, Y_train, batch_size=batchsize) 
validation_generator=validationdatagenerator.flow(X_test, Y_test,batch_size=batchsize)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

history= model.fit_generator(train_generator, steps_per_epoch=int(len(X_train)/batchsize), 
                    epochs=10, validation_data=validation_generator, 
                    validation_steps=int(len(X_test)/batchsize))

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('model_accuracy.png')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.savefig('model_loss.png')

score = model.evaluate(X_test, Y_test, verbose=0)
print(score)

# Model to JSON
model_json = model.to_json()
with open("model_project_work.json", "w") as json_file:
    json_file.write(model_json)
# Weights to HDF5
model.save("model.h5")
print("Saved model to disk")

print(X_train.shape)
print(X_test[0].shape)
y_pred = []
p=100
for i in range (200):
    image = X_train[i]
    imagematrix =  np.asarray(image)
    #img1=imagematrix.reshape(img_rows,img_cols,3)
    #plt.imshow(img1)
    #print(imagematrix.shape)
    imagepredict = np.expand_dims(imagematrix, axis=0)
    #print(imagepredict.shape)
    y_pred1 = model.predict(imagepredict)
    y_pred2 = [x * p for x in y_pred1]
    y_pred3 = np.max(y_pred2)
    y_pred.append(y_pred3)
    print(y_pred3)
y_pred = np.asarray(y_pred)

Y_val = trainLabels['level']
y_pred =  model.predict_generator(validation_generator)

threshold = 0.5
y_final = np.where(y_pred > threshold, 1,0)

y_final.size

# Generate a classification report
report = classification_report(Y_val, y_final, target_names=['0','1', '2','3','4'])

print(report)

from sklearn.metrics import confusion_matrix
# Predict the values from the validation dataset

# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_val, y_pred) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()