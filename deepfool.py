import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.models import model_from_json
from keras import backend as K
import numpy as np
from PIL import Image

eta = 0.02
clip_min = 0
clip_max = 1
epochs = 100
min_prob = 0.1

img_path = 'images_cropped2/00000/00000_00000.ppm'
model_path = 'DNN.json'
weights_path = 'DNN_weights.h5'

def _cond(i, z):
    # print(str(i))
    xadv = tf.clip_by_value(x + z * (1 + eta), clip_min, clip_max)
    y = tf.reshape(model(xadv), [-1])
    p = tf.reduce_max(y)
    k = tf.argmax(y)
    return tf.logical_and(tf.less(i, epochs), tf.logical_or(tf.equal(tf.cast(true_class, dtype=tf.int64), k), tf.less(p, min_prob)))
    # return tf.logical_or(tf.equal(tf.cast(true_class, dtype=tf.int64), k), tf.less(p, min_prob))

def _body(i, z):
    # perturbate the img
    xadv = tf.clip_by_value(x + z*(1+eta), clip_min, clip_max)

    # get current class probabilities
    labels = model(xadv)[0]

    # calculate gradients for each class
    gs = [tf.reshape(tf.gradients(labels[i], xadv)[0], [-1])
          for i in range(num_classes)]

    # merge results into a single tensor
    g = tf.stack(gs, axis=0)

    # separete true class label and gradient from the other labels and gradients
    yk, yo = labels[true_class], tf.concat((labels[:true_class], labels[(true_class + 1):]), axis=0)
    gk, go = g[true_class], tf.concat((g[:true_class], g[(true_class + 1):]), axis=0)

    # reshape to not include the true class
    yo.set_shape(num_classes - 1)
    go.set_shape([num_classes - 1, xflat])

    # mathemagic - comments are corresponding (pytorch) code from the original paper repository
    # w_k = cur_grad - grad_orig
    a = tf.abs(yo - yk)
    # f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()
    b = go - gk
    # pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())
    c = tf.norm(b, axis=1)
    score = a / c

    # pick the closest direction
    # score == pert_k in orig
    ind = tf.argmin(score)

    si, bi = score[ind], b[ind]
    # Added 1e-4 for numerical stability
    dx = (si+1e-4) * bi
    dx = tf.reshape(dx, [-1] + x.get_shape().as_list()[1:])
    return i + 1, z + dx

# get the img
img = image.img_to_array(image.load_img(img_path))
img /= 255
img = np.reshape(img, newshape=(40, 40, 3))
img_in = np.expand_dims(img, axis=0)

# print('img: ' +str(img))

# load model
json_file = open(model_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(weights_path)

# get the original predicted class
_preds = model.predict(img_in)[0]
true_class = list(_preds).index(np.amax(_preds))
print('Original class: ' + str(true_class))
num_classes = len(_preds)

x = tf.Variable(img_in, dtype='float32')
xflat = len(img.flatten())

num_it, noise = tf.while_loop(_cond, _body, [0, tf.zeros_like(x)], back_prop=False)

# xadv = tf.stop_gradient(
xadv = tf.stop_gradient(x +  tf.clip_by_value(noise*(1+eta), clip_min, clip_max))
# xadv = tf.clip_by_value(xadv, clip_min, clip_max)

# noise scaling
_min = tf.cast(tf.reduce_min(noise), dtype=tf.float32)
_max = tf.cast(tf.reduce_max(noise), dtype=tf.float32)

noise = tf.cast(noise, dtype=tf.float32)
scaled_noise = tf.map_fn(lambda x: ((x - _min) / abs(_max - _min)) * 255, noise)

sess = tf.Session()
with sess.as_default():
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # pred = model(x)[0]
    # preds = sess.run(pred)
    # print(str(preds.dtype))

    xadv = sess.run([xadv])
    xadv = xadv[0] * 255
    xadv = xadv.reshape(40, 40, 3)

    # print(str(xadv.shape))
    print('Finished after ' + str(num_it.eval()) + ' iterations')

    scaled_noise = sess.run(scaled_noise)[0]
    # print(str(scaled_noise.shape))

    # Save the scaled up noise
    im = Image.fromarray(scaled_noise.astype(np.uint8))
    im.save('deepfool-sc-noise.png')

    pert = np.sum(np.absolute(noise.eval()))

    print('pert: ' + str(pert))

    # Save the hacked image!
    im = Image.fromarray(xadv.astype(np.uint8))
    im.save('deepfool-img.png')

    print('noise max: ' + str(_max.eval()))
    print('noise min: ' + str(_min.eval()))
    # print('sc_noise: ' + str(scaled_noise))
    # print('noise: ' + str(noise.eval()))
    # xadv2 = img + (noise.eval() * (1+eta))
    # print('xadv2: ' + str(xadv2*255))
    # print('orig: ' + str(img))
    # print('x: ' + str(x.eval()))
    # print('noise: ' + str(noise.eval()))
    # print('xadv: ' + str(xadv))
