#######################################################################
#Imports and Constants
######################################################################
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.models import model_from_json



# Training Params
num_steps = 70000
batch_size = 128
learning_rate = 0.0002

# Dims and topology
IMG_DIM = 4800 # 40*40*3
GEN_HID_UN1 = 256
GEN_HID_UN2 = 256
DIS_HID_UN1 = 256
DIS_HID_UN2 = 256
NOISE_DIM = 100

def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

# Store layers weight & bias
weights = {
    'gen_hidden1': tf.Variable(glorot_init([NOISE_DIM, GEN_HID_UN1])),
    'gen_hidden2': tf.Variable(glorot_init([NOISE_DIM, GEN_HID_UN2])),
    'gen_out': tf.Variable(glorot_init([GEN_HID_UN1, IMG_DIM])),
    'disc_hidden1': tf.Variable(glorot_init([IMG_DIM, DIS_HID_UN1])),
    'disc_hidden2': tf.Variable(glorot_init([IMG_DIM, DIS_HID_UN2])),
    'disc_out': tf.Variable(glorot_init([DIS_HID_UN1, 1])),
}
biases = {
    'gen_hidden1': tf.Variable(tf.zeros([GEN_HID_UN1])),
    'gen_hidden2': tf.Variable(tf.zeros([GEN_HID_UN2])),
    'gen_out': tf.Variable(tf.zeros([IMG_DIM])),
    'disc_hidden1': tf.Variable(tf.zeros([DIS_HID_UN1])),
    'disc_hidden2': tf.Variable(tf.zeros([DIS_HID_UN2])),
    'disc_out': tf.Variable(tf.zeros([1])),
}

# Generator
def generator(x):
    hidden_layer1 = tf.matmul(x, weights['gen_hidden1'])
    hidden_layer1 = tf.add(hidden_layer1, biases['gen_hidden1'])
    hidden_layer1 = tf.nn.relu(hidden_layer1)
    hidden_layer2 = tf.matmul(hidden_layer1, weights['gen_hidden2'])
    hidden_layer2 = tf.add(hidden_layer2, biases['gen_hidden2'])
    hidden_layer2 = tf.nn.relu(hidden_layer2)
    out_layer = tf.matmul(hidden_layer2, weights['gen_out'])
    out_layer = tf.add(out_layer, biases['gen_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

# Discriminator
def discriminator(x):
    hidden_layer1 = tf.matmul(x, weights['disc_hidden1'])
    hidden_layer1 = tf.add(hidden_layer1, biases['disc_hidden1'])
    hidden_layer1 = tf.nn.relu(hidden_layer1)
    hidden_layer2 = tf.matmul(hidden_layer1, weights['disc_hidden2'])
    hidden_layer2 = tf.add(hidden_layer2, biases['disc_hidden2'])
    hidden_layer2 = tf.nn.relu(hidden_layer2)
    out_layer = tf.matmul(hidden_layer2, weights['disc_out'])
    out_layer = tf.add(out_layer, biases['disc_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

def __main__():
    train_datagen = IDG.flow_from_directory(
            directory=ORIGINDIR, target_size = (40,40), classes=signs_classes, batch_size=BATCH)

    # Build Networks
    # Network Inputs
    gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
    disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name='disc_input')

    # Build Generator Network
    gen_sample = generator(gen_input)

    # Build 2 Discriminator Networks (one from noise input, one from generated samples)
    disc_real = discriminator(disc_input)
    disc_fake = discriminator(gen_sample)

    # Build Loss
    gen_loss = -tf.reduce_mean(tf.log(disc_fake))
    disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

    # Build Optimizers
    optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Training Variables for each optimizer
    # By default in TensorFlow, all variables are updated by each optimizer, so we
    # need to precise for each one of them the specific variables to update.
    # Generator Network Variables
    gen_vars = [weights['gen_hidden1'], weights['gen_hidden2'], weights['gen_out'],
                biases['gen_hidden1'], biases['gen_hidden2'], biases['gen_out']]
    # Discriminator Network Variables
    disc_vars = [weights['disc_hidden1'], weights['disc_hidden2'],weights['disc_out'],
                biases['disc_hidden1'], biases['disc_hidden2'], biases['disc_out']]

    # Create training operations
    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start Training
    # Start a new TF session
    sess = tf.Session()

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)
        # Generate noise to feed to the generator
        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

        # Train
        feed_dict = {disc_input: batch_x, gen_input: z}
        _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                feed_dict=feed_dict)
        if i % 2000 == 0 or i == 1:
            print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))


    # used image
    IMG_PATH = 'images_orig/00000/00000_00000_cropped.ppm'


    # load json and create model
    json_file = open('DNN.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("DNN_weights.h5")
    print("Loaded model from disk")

    img = image.load_img(IMG_PATH, target_size=(40, 40))
    input_image = image.img_to_array(img)

    # standartise
    input_image /= 255.

    # add batch size dim
    input_image = np.expand_dims(input_image, axis=0)

    predictions = model.predict(input_image)

    # Convert the predictions into text and print them
    class_id = np.where(predictions == np.amax(predictions))[0][0]
    print(str(class_id) + ' class predicted with confidence of ' + str(predictions[0][class_id]))

if __name__ == '__main__':
    main()