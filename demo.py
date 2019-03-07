##########################################################################################
#Imports and Consts
##########################################################################################
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.models import model_from_json
import matplotlib.pyplot as plt

##########################################################################################
#functionality
##########################################################################################
def load_DNN():
    # load json and create model
    json_file = open('DNN.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("DNN_weights.h5")
    print("Loaded model from disk")
    return model

def test_image(img_path, model, view_img=False):
    img = image.load_img(img_path, target_size=(40, 40))
    input_image = image.img_to_array(img)

    # standartise
    input_image /= 255.

    #view the given image
    if(view_img):
        plt.imshow(input_image)
        plt.show()

    # add batch size dim
    input_image = np.expand_dims(input_image, axis=0)

    predictions = model.predict(input_image)
    print(predictions)

def test_directory(dir_path, model):
    print("todo")
    #use os to find all files in the sub/directories
    #feed files into test_image

##########################################################################################
#example
##########################################################################################
def __main__():
    img_path='images_cropped/00000/00000_00000.ppm'
    model = load_DNN()
    test_image(img_path, model, True)
    
if __name__ =='__main__':
    __main__()

