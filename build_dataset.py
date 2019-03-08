# code mostly taken from https://www.youtube.com/watch?v=bqeUmLCgsVw
from random import shuffle
import os
import sys
import cv2 as cv
import glob
import tensorflow as tf

ROOT_DIR = 'images_cropped'
DATASET_FILE_NAME = 'data'

def _int64_feature(val):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))
def _bytes_feature(val):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))

def create_data_record(out_file_name, addrs, labels):
    writer = tf.python_io.TFRecordWriter(out_file_name)
    print('Loading the images in')
    for addr, label in zip(addrs, labels):
        print(addr)
        img = cv.imread(addr)
        if img is None:
            continue

        # create feature
        feature = {
            'image_raw': _bytes_feature(img.tostring()),
            'label': _int64_feature(label)
        }

        # save the record to file
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

def __main__():
    # get all the img addresses
    path = ROOT_DIR + '/*/*.ppm'
    addrs = glob.glob(path)

    # get the labels
    labels = [int(addr.split('\\')[1]) for addr in addrs]

    # shuffle
    _c = list(zip(addrs, labels))
    shuffle(_c)
    addrs, labels = zip(*_c)

    create_data_record(DATASET_FILE_NAME + '.tfrecords', addrs, labels)

# def load_data(data_directory):
#     directories = [d for d in os.listdir(data_directory)
#                    if os.path.isdir(os.path.join(data_directory, d))]
#     labels = []
#     images = []
#     for d in directories:
#         label_directory = os.path.join(data_directory, d)
#         file_names = [os.path.join(label_directory, f)
#                       for f in os.listdir(label_directory)
#                       if f.endswith(".ppm")]
#         for f in file_names:
#             img = Image.open(f)
#             pixels = list(img.getdata())  # extracts the pixels
#             width, height = img.size
#             images.append([pixels[i * width:(i + 1) * width] for i in range(height)])
#             labels.append(int(d))
#     return images, labels

if __name__ == '__main__':
    __main__()