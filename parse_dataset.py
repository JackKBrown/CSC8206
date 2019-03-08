import tensorflow as tf
import numpy as np

def parser(rec):
    keys_to_features = {
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(rec, keys_to_features)
    img = tf.decode_raw(parsed['image_raw'], tf.uint8)
    img = tf.cast(img, tf.float32)
    img = tf.reshape(img, shape=[40, 40, 3])
    label = tf.cast(parsed['label'], tf.int32)

    return img, label

def input_fn(filenames, train, batch_size=32, buffer_size=2048):
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(parser)

    if train:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        num_repeat = None
    else:
        num_repeat = 1

    dataset = dataset.repeat(num_repeat)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    imgs_batch, labels_batch = iterator.get_next()

    x = {'image', imgs_batch}
    y = labels_batch

    return x, y

def input_wrap_fn():
    return input_fn(['data.tfrecords'], False)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
features, labels = input_wrap_fn()

sess.run(train_iterator.initializer)
x, y = sess.run([features['image'], labels])
print(x.shape, y.shape)


print(str(x))