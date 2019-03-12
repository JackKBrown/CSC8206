import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import argparse

parser = argparse.ArgumentParser(description='Predict img class using saved DNN')
parser.add_argument('image_path', type=str, help='Image path')

args = parser.parse_args()

if not args.image_path.strip():
    #IMG_LOCATION = 'hacked-image.png'
    IMG_LOCATION = 'images_cropped/00000/00000_00000.ppm'
    print('Using default image: ' + IMG_LOCATION)
else:
    IMG_LOCATION = args.image_path.strip()

# get the pixels
im = image.load_img(IMG_LOCATION, target_size=(40,40)) # open the image
pixels = image.img_to_array(im) # extracts the pixels
pixels /= 255.
pixels = np.reshape(pixels, newshape=(40, 40, 3))
pixels = np.expand_dims(pixels, axis=0)
# print(str(pixels))
# width, height = im.size
# pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
# print(len(pixels))

# load json and create model
json_file = open('DNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("DNN_weights.h5")
#print("Loaded model from disk")
preds = model.predict(pixels)[0]
# print(str(pixels))
np.set_printoptions(precision=4, suppress=True)
print(str(preds*100))
print('Class id: ' + str(list(preds).index(np.amax(preds))))
