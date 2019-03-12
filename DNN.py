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
#this is only here as it is the cleanest way to split both labels and features
from sklearn.model_selection import train_test_split

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

#ls -l images_cropped/*/* | egrep -c '^-' returns 39209
BATCH=39209



#######################################################################
#Data preperation
######################################################################
def data_prep():
    #IDG lets you pull images from a directory and label them according to the subdir
    #see keras ImageDataGenerator parameters for more customisability
    IDG = ImageDataGenerator(rescale=1./255, dtype='float32')

    train_datagen = IDG.flow_from_directory( directory=ORIGINDIR, target_size = (40,40), classes=signs_classes, batch_size=BATCH)
    #fetch a batch of images and labels
    #images, labels = next(train_datagen)
    #test_images, test_labels = next(train_datagen)
    
    data, labels = next(train_datagen)
    
    images, test_images, labels, test_labels=train_test_split(data, labels, test_size=0.20)
    
    print("shape of the image files")
    print(images.shape)
    return images, test_images, labels, test_labels


#######################################################################
#DNN model
######################################################################
def define_model(num_nodes):
    #Define the DNN model used
    model = Sequential()
    
    #input is a 40 by 40 rgb image
    model.add(Flatten(input_shape=(40, 40, 3)))
    #hidden layers
    model.add(Dense(num_nodes, activation='relu'))
    model.add(Dense(num_nodes, activation='relu'))
    model.add(Dense(num_nodes, activation='relu'))
    #output
    model.add(Dense(len(signs_classes), activation='softmax'))
    
    return model

def compile_and_fit(images, labels, test_images, test_labels, model, num_epochs):
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(images, labels, batch_size=32, epochs=num_epochs)
    score = model.evaluate(test_images, test_labels, batch_size=32)
    print(score)
    return score

def __main__():
    images, test_images, labels, test_labels = data_prep()
    model = define_model(256)
    compile_and_fit(images, labels, test_images, test_labels, model, 10)

    #save the model
    model_json = model.to_json()
    with open("DNN.json", "w") as json_file:
        json_file.write(model_json)
    #serialize weights to HDF5
    model.save_weights("DNN_weights.h5")
    print("Saved model to disk")

if __name__ =='__main__':
    __main__()

