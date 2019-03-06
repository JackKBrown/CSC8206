#######################################################################
#Imports and Constants
######################################################################
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.optimizers import SGD

#this is the file containing the image data
ORIGINDIR='images_cropped/'

#array of all the class dir names in ORIGINDIR
signs_classes=['00000','00001','00002','00003','00004',
        '00005','00006','00007','00008','00009',
        '00010','00011','00012','00013','00014',
        '00015','00016','00017','00018','00019',
        '00020','00021','00022','00023','00024',
        '00025','00026','00027','00028','00029',
        '00030','00031','00032','00033','00034',
        '00035','00036','00037','00038','00039',
        '00040','00041','00042']

BATCH=19000



#######################################################################
#Data preperation
######################################################################

#IDG lets you pull images from a directory and label them according to the subdir
#see keras ImageDataGenerator parameters for more customisability
IDG = ImageDataGenerator( rescale=1./255)

train_datagen = IDG.flow_from_directory(
        directory=ORIGINDIR, target_size = (40,40), classes=signs_classes, batch_size=BATCH)


#fetch a batch of images and labels
images, labels = next(train_datagen)
test_images, test_labels = next(train_datagen)
print(images.shape)

print(labels)


#######################################################################
#DNN model
######################################################################

#Define the DNN model used
model = Sequential()

#input is a 40 by 40 rgb image
model.add(Flatten(input_shape=(40, 40, 3)))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
#dropout?
model.add(Dense(43, activation='softmax'))

#these two lines will show a sample image from the dataset
#plt.imshow(images[0])
#plt.show()

def compile_and_fit(images, labels, test_images, test_labels, model):
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(images, labels, batch_size=32, epochs=10)
    score = model.evaluate(test_images, test_labels, batch_size=32)
    print(score)

compile_and_fit(images, labels, test_images, test_labels, model)

model_json = model.to_json()
with open("DNN.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("DNN_weights.h5")
print("Saved model to disk")
