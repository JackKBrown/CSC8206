import tensorflow as tf
from df_orig import deepfool
from keras.preprocessing import image
from PIL import Image
from keras.models import model_from_json
from keras import backend as K
import numpy as np
import argparse

class Dummy:
    pass



parser = argparse.ArgumentParser(description='Craft adversarial examples')
parser.add_argument('--img',dest='image_path', type=str, default='images_cropped/00000/00000_00000.ppm', help='Image path')
parser.add_argument('--cost',dest='min_cost', type=float, default=0.90, help='Minimum certaintity to stop')
parser.add_argument('--save',dest='save_as', type=str, default='deepfooled-img.png', help='Where to save the perturbated image')
parser.add_argument('--eta',dest='eta', type=float, default=0.01, help='Eta (Learning rate)')
parser.add_argument('--noise',dest='noise', type=str, default='deepfool-noise.png', help='Where to save the noise')
parser.add_argument('--noises',dest='noises', type=str, default='deepfool-noise_scaled.png', help='Where to save the scaled noise')

args = parser.parse_args()

sess = tf.InteractiveSession()
K.set_session(sess)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

env = Dummy()

# load json and create model
json_file = open('DNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("DNN_weights.h5")

# get the pixels
im = image.load_img(args.image_path, target_size=(40,40)) # open the image
x = image.img_to_array(im) # extracts the pixels
x /= 255.
x = np.reshape(x, newshape=(40, 40, 3))
x = np.expand_dims(x, axis=0)

env.adv_epochs = tf.placeholder(tf.int32, (), name='adv_epochs')
env.x = tf.placeholder(tf.float32, (None, 40, 40, 3), name='x')

xadv = deepfool(model, env.x, eta=args.eta, epochs=3, batch=False, clip_min=0.0, clip_max=1.0, min_prob=0.0)

adv = sess.run(xadv, feed_dict={env.x: x, env.adv_epochs: 3})

# print(adv.shape)
# print(adv)
print(model.predict(adv))

_preds = model.predict(adv)[0]
true_class = list(_preds).index(np.amax(_preds))
print('New class: ' + str(true_class))

adv = adv[0]
noise = x[0] - adv

adv *= 255
# Save the hacked image!
im = Image.fromarray(adv.astype(np.uint8))
im.save(args.save_as)

# Save the hacked image!
im = Image.fromarray(noise.astype(np.uint8))
im.save(args.noise)

_min = np.amin(noise)
_max = np.amax(noise)
scaled_noise = np.array(list(map(lambda x: ((x -_min) / abs(_max  - _min)) * 255, noise)))

im = Image.fromarray(scaled_noise.astype(np.uint8))
im.save(args.noises)

pert = np.sum(np.abs(noise))
print('Total pert: ' + str(pert))
sess.close()

