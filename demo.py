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
    return predictions

def test_directory(dir_path, model):
    print("todo")
    #use os to find all files in the sub/directories
    #feed files into test_image

##########################################################################################
#example
##########################################################################################
def __main__():
    #make predictions for image 00000/00000_00000.ppm
    model = load_DNN()
    img_path='images_cropped/00000/00000_00000.ppm'
    unalt_pred = test_image(img_path, model)
    img_path='00000Demo/z.ppm'
    rand_pred = test_image(img_path, model)
    img_path='00000Demo/masked_opaque.ppm'
    mask_pred = test_image(img_path, model)
    img_path='00000Demo/masked_transparency.ppm'
    maskt_pred = test_image(img_path, model)

    #find max value prediction
    #unalt_class = np.where(unalt_pred == np.amax(unalt_pred))[0][0]
    unalt_class=unalt_pred.argmax()
    rand_class=rand_pred.argmax()
    mask_class=mask_pred.argmax()
    maskt_class=maskt_pred.argmax()
    print(mask_pred.argmax())
    print(unalt_pred)
    print(unalt_pred.max())
    print(str("%.8f" %unalt_pred[0][0]))

    #print off nice looking table
    print('________________________________________________________________________')
    print('|      unalt      |       rand      |       mask      |      maskt     |')
    print('________________________________________________________________________')
    
    #print off the expected class values
    print('| 0 at ' + str(unalt_pred[0][0]) +' | 0 at ' + str(rand_pred[0][0])+' | 0 at '+str(mask_pred[0][0])+' | 0 at '+str(maskt_pred[0][0])+' |')
    
    #print off the predicted class values
    print('| '+str(unalt_class) + ' at ' + str(unalt_pred[0][unalt_class]) +' | '+ str(rand_class) + ' at ' + str(rand_pred[0][rand_class])+' | '+str(mask_class)+' at '+str(mask_pred[0][mask_class])+' | '+str(maskt_class)+' at '+str(maskt_pred[0][maskt_class])+' |')
    #todo
    #show images all at the end
    #get output into a nice table
    
if __name__ =='__main__':
    __main__()

