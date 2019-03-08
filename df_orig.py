import tensorflow as tf
from PIL import Image
from keras.preprocessing import image
import numpy as np
from keras.models import model_from_json
from keras import backend as K

__all__ = ['deepfool']


def deepfool(model, x, noise=False, eta=0.01, epochs=3, batch=False,
             clip_min=0.0, clip_max=1.0, min_prob=0.0):
    """DeepFool implementation in Tensorflow.
    The original DeepFool will stop whenever we successfully cross the
    decision boundary.  Thus it might not run total epochs.  In order to force
    DeepFool to run full epochs, you could set batch=True.  In that case the
    DeepFool will run until the max epochs is reached regardless whether we
    cross the boundary or not.  See https://arxiv.org/abs/1511.04599 for
    details.
    :param model: Model function.
    :param x: 2D or 4D input tensor.
    :param noise: Also return the noise if True.
    :param eta: Small overshoot value to cross the boundary.
    :param epochs: Maximum epochs to run.
    :param batch: If True, run in batch mode, will always run epochs.
    :param clip_min: Min clip value for output.
    :param clip_max: Max clip value for output.
    :param min_prob: Minimum probability for adversarial samples.
    :return: Adversarials, of the same shape as x.
    """
    y = tf.stop_gradient(model(x))

    fns = [[_deepfool2, _deepfool2_batch], [_deepfoolx, _deepfoolx_batch]]

    i = int(y.get_shape().as_list()[1] > 1)
    j = int(batch)
    fn = fns[i][j]

    if batch:
        delta = fn(model, x, eta=eta, epochs=epochs, clip_min=clip_min,
                   clip_max=clip_max)
    else:
        def _f(xi):
            xi = tf.expand_dims(xi, axis=0)
            z = fn(model, xi, eta=eta, epochs=epochs, clip_min=clip_min,
                   clip_max=clip_max, min_prob=min_prob)
            return z[0]

        delta = tf.map_fn(_f, x, dtype=(tf.float32), back_prop=False,
                          name='deepfool')

    if noise:
        return delta

    xadv = tf.stop_gradient(x + delta*(1+eta))
    xadv = tf.clip_by_value(xadv, clip_min, clip_max)
    return xadv


def _prod(iterable):
    ret = 1
    for x in iterable:
        ret *= x
    return ret


def _deepfool2(model, x, epochs, eta, clip_min, clip_max, min_prob):
    """DeepFool for binary classifiers.
    Note that DeepFools that binary classifier outputs +1/-1 instead of 0/1.
    """
    y0 = tf.stop_gradient(tf.reshape(model(x), [-1])[0])
    y0 = tf.to_int32(tf.greater(y0, 0.0))

    def _cond(i, z):
        xadv = tf.clip_by_value(x + z*(1+eta), clip_min, clip_max)
        y = tf.stop_gradient(tf.reshape(model(xadv), [-1])[0])
        y = tf.to_int32(tf.greater(y, 0.0))
        return tf.logical_and(tf.less(i, epochs), tf.equal(y0, y))

    def _body(i, z):
        xadv = tf.clip_by_value(x + z*(1+eta), clip_min, clip_max)
        y = tf.reshape(model(xadv), [-1])[0]
        g = tf.gradients(y, xadv)[0]
        dx = - y * g / (tf.norm(g) + 1e-10)  # off by a factor of 1/norm(g)
        return i+1, z+dx

    _, noise = tf.while_loop(_cond, _body, [0, tf.zeros_like(x)],
                             name='_deepfool2', back_prop=False)
    return noise


def _deepfool2_batch(model, x, epochs, eta, clip_min, clip_max):
    """DeepFool for binary classifiers in batch mode.
    """
    xshape = x.get_shape().as_list()[1:]
    dim = _prod(xshape)

    def _cond(i, z):
        return tf.less(i, epochs)

    def _body(i, z):
        xadv = tf.clip_by_value(x + z*(1+eta), clip_min, clip_max)
        y = tf.reshape(model(xadv), [-1])
        g = tf.gradients(y, xadv)[0]
        n = tf.norm(tf.reshape(g, [-1, dim]), axis=1) + 1e-10
        d = tf.reshape(-y / n, [-1] + [1]*len(xshape))
        dx = g * d
        return i+1, z+dx

    _, noise = tf.while_loop(_cond, _body, [0, tf.zeros_like(x)],
                             name='_deepfool2_batch', back_prop=False)
    return noise


def _deepfoolx(model, x, epochs, eta, clip_min, clip_max, min_prob):
    """DeepFool for multi-class classifiers.
    Assumes that the final label is the label with the maximum values.
    """
    y0 = tf.stop_gradient(model(x))
    y0 = tf.reshape(y0[0], [-1])
    k0 = tf.argmax(y0)

    ydim = y0.get_shape().as_list()[0]
    xdim = x.get_shape().as_list()[1:]
    xflat = _prod(xdim)

    def _cond(i, z):
        xadv = tf.clip_by_value(x + z*(1+eta), clip_min, clip_max)
        y = tf.reshape(model(xadv), [-1])
        p = tf.reduce_max(y)
        k = tf.argmax(y)
        return tf.logical_and(tf.less(i, epochs),
                              tf.logical_or(tf.equal(k0, k),
                                            tf.less(p, min_prob)))

    def _body(i, z):
        xadv = tf.clip_by_value(x + z*(1+eta), clip_min, clip_max)
        y = tf.reshape(model(xadv), [-1])

        gs = [tf.reshape(tf.gradients(y[i], xadv)[0], [-1])
              for i in range(ydim)]
        g = tf.stack(gs, axis=0)

        yk, yo = y[k0], tf.concat((y[:k0], y[(k0+1):]), axis=0)
        gk, go = g[k0], tf.concat((g[:k0], g[(k0+1):]), axis=0)

        yo.set_shape(ydim - 1)
        go.set_shape([ydim - 1, xflat])

        a = tf.abs(yo - yk)
        b = go - gk
        c = tf.norm(b, axis=1)
        score = a / c
        ind = tf.argmin(score)

        si, bi = score[ind], b[ind]
        dx = si * bi
        dx = tf.reshape(dx, [-1] + xdim)
        return i+1, z+dx

    _, noise = tf.while_loop(_cond, _body, [0, tf.zeros_like(x)],
                             name='_deepfoolx', back_prop=False)
    return noise


def _deepfoolx_batch(model, x, epochs, eta, clip_min, clip_max):
    """DeepFool for multi-class classifiers in batch mode.
    """
    y0 = tf.stop_gradient(model(x))
    B, ydim = tf.shape(y0)[0], y0.get_shape().as_list()[1]

    k0 = tf.argmax(y0, axis=1, output_type=tf.int32)
    k0 = tf.stack((tf.range(B), k0), axis=1)

    xshape = x.get_shape().as_list()[1:]
    xdim = _prod(xshape)

    perm = list(range(len(xshape) + 2))
    perm[0], perm[1] = perm[1], perm[0]

    def _cond(i, z):
        return tf.less(i, epochs)

    def _body(i, z):
        xadv = tf.clip_by_value(x + z*(1+eta), clip_min, clip_max)
        y = model(xadv)

        gs = [tf.gradients(y[:, i], xadv)[0] for i in range(ydim)]
        g = tf.stack(gs, axis=0)
        g = tf.transpose(g, perm)

        yk = tf.expand_dims(tf.gather_nd(y, k0), axis=1)
        gk = tf.expand_dims(tf.gather_nd(g, k0), axis=1)

        a = tf.abs(y - yk)
        b = g - gk
        c = tf.norm(tf.reshape(b, [-1, ydim, xdim]), axis=-1)

        # Assume 1) 0/0=tf.nan 2) tf.argmin ignores nan
        score = a / c

        ind = tf.argmin(score, axis=1, output_type=tf.int32)
        ind = tf.stack((tf.range(B), ind), axis=1)

        si, bi = tf.gather_nd(score, ind), tf.gather_nd(b, ind)
        si = tf.reshape(si, [-1] + [1]*len(xshape))
        dx = si * bi
        return i+1, z+dx

    _, noise = tf.while_loop(_cond, _body, [0, tf.zeros_like(x)],
                             name='_deepfoolx_batch', back_prop=False)
    return noise


# class Dummy:
#     pass
#
#
# env = Dummy()
# eta = 0.04
# clip_min = 0
# clip_max = 1
# epochs = 10000000
# min_prob = 0.1
#
# img_path = 'images_cropped2/00000/00000_00000.ppm'
# model_path = 'DNN.json'
# weights_path = 'DNN_weights.h5'
#
# # get the img
# img = image.img_to_array(image.load_img(img_path))
# img /= 255
# img = np.reshape(img, newshape=(40, 40, 3))
# img_in = np.expand_dims(img, axis=0)
#
# # print('img: ' +str(img))
#
# # load model
# json_file = open(model_path, 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # load weights into new model
# model.load_weights(weights_path)
#
# with tf.variable_scope('model', reuse=True):
#     env.x = tf.placeholder(tf.float32, (None, 40, 40, 3), name='x')
#     env.adv_epochs = tf.placeholder(tf.int32, (), name='adv_epochs')
#     env.xadv = deepfool(model, env.x, epochs=env.adv_epochs)
#
#
# # x = tf.Variable(img_in, dtype='float32')
# x = img_in
#
# sess = tf.InteractiveSession()
# K.set_session(sess)
# sess.run(tf.global_variables_initializer())
# sess.run(tf.local_variables_initializer())
#
# xadv = sess.run(env.xadv, feed_dict={env.x: x, env.adv_epochs: epochs})
#
# xadv = xadv[0] * 255
# xadv = xadv.reshape(40, 40, 3)
#
# print(str(xadv))
#
# im = Image.fromarray(xadv.astype(np.uint8))
# im.save('deepfool-img.png')