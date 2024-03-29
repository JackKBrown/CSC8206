# code mostly taken from https://medium.com/@ageitgey/machine-learning-is-fun-part-8-how-to-intentionally-trick-neural-networks-b55da32b7196
import math
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications import inception_v3
from keras import backend as K
import tensorflow as tf
from PIL import Image
import argparse

def __main__():
    parser = argparse.ArgumentParser(description='Craft adversarial examples')
    parser.add_argument('--img',dest='image_path', type=str, default='images_cropped/00000/00000_00000.ppm', help='Image path')
    parser.add_argument('--target',dest='target_cls', type=int, default=-1, help='Target class, -1 for minimizing the original class')
    parser.add_argument('--cost',dest='min_cost', type=float, default=0.90, help='Minimum certaintity to stop')
    parser.add_argument('--clip',dest='clip_range', type=float, default=0.1, help='How much change to allow [0,1]')
    parser.add_argument('--learn-rate',dest='learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--save',dest='save_as', type=str, default='hacked-img.png', help='Where to save the perturbated image')
    parser.add_argument('--noise',dest='noise', type=str, default='', help='Where to save the noise')
    parser.add_argument('--noises',dest='noises', type=str, default='', help='Where to save the scaled noise')
    parser.add_argument('--sign',dest='sign', action='store_true', help='If set, sign(gradients) is used rather than gradients')

    args = parser.parse_args()

    # used image
    IMG_PATH = args.image_path

    # load json and create model
    json_file = open('DNN.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("DNN_weights.h5")
    print("Loaded model from disk")

    # Grab a reference to the first and last layer of the neural net
    model_input_layer = model.layers[0].input
    model_output_layer = model.layers[-1].output

    # Load the image to hack
    original_image = image.img_to_array(image.load_img(IMG_PATH, target_size=(40, 40)))
    img = original_image.copy()
    #
    # im = Image.fromarray(img.astype(np.uint8))
    # im.save(args.save_as)

    # Scale the image so all pixel intensities are between [0, 1] as the model expects
    img /= 255.

    # Add a 4th dimension for batch size (as Keras expects)
    img = np.expand_dims(img, axis=0)

    # id of the target class = class predicted on original img
    _pred = model.predict(img)[0]
    true_class = list(_pred).index(np.amax(_pred))
    print('Original class: ' + str(true_class))

    # Pre-calculate the maximum change we will allow to the image
    # We'll make sure our hacked image never goes past this so it doesn't look funny.
    # A larger number produces an image faster but risks more distortion.
    max_change_above = img + args.clip_range
    max_change_below = img - args.clip_range

    # Create a copy of the input image to hack on
    hacked_image = np.copy(img)

    # imgh = np.round(hacked_image * 255.)
    # # Save the hacked image!
    # im = Image.fromarray(imgh.astype(np.uint8))
    # im.save(args.save_as)
    # print('Hacked image save as ' + args.save_as)

    # How much to update the hacked image in each iteration
    learning_rate = abs(args.learning_rate)

    # Define the cost function.
    if args.target_cls >= 0:
        # Our cost will be the likelihood of the image being the target class
        cost_function = model_output_layer[0, args.target_cls]
    else:
        # Our 'cost' will be the likelihood out image is not the original (ground truth) class
        cost_function = 1 - model_output_layer[0, true_class]

    # We'll ask Keras to calculate the gradient based on the input image and the currently predicted class
    # In this case, referring to "model_input_layer" will give us back image we are hacking.
    gradient_function = K.gradients(cost_function, model_input_layer)[0]

    # Create a Keras function that we can call to calculate the current cost and gradient
    grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()], [cost_function, gradient_function])

    cost = 0.0
    i = 0
    cost_dif_threshold = 1e-6
    last_cost_cp = 0.0
    cost_cp_freq = 500

    # In a loop, keep adjusting the hacked image slightly so that it tricks the model more and more
    # until it gets to given confidence
    while cost < abs(args.min_cost):
        # Check how close the image is to our target class and grab the gradients we
        # can use to push it one more step in that direction.
        # Note: It's really important to pass in '0' for the Keras learning mode here!
        # Keras layers behave differently in prediction vs. train modes!
        cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])

        # print('min, max gradients: ' + str(np.amin(gradients)) + ', ' + str(np.amax(gradients)))
        
        # Move the hacked image one step further towards fooling the model
        if args.sign:
            hacked_image += np.sign(gradients) * learning_rate
        else:
            hacked_image += gradients * learning_rate

        # Ensure that the image doesn't ever change too much to either look funny or to become an invalid image
        hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
        hacked_image = np.clip(hacked_image, 0, 1.0)

        _pred = model.predict(hacked_image)[0]
        cur_pred = list(_pred).index(np.amax(_pred))
        __pred = _pred * 100
        np.set_printoptions(precision=4, suppress=True)
        # print(__pred)

        if args.target_cls < 0:
            print(str(i) + ": Likelihood that the image is not the original class: {:.8}%".format(cost * 100))
        else:
            print(str(i) + ': Likelihood that the image is the target class: {:.8}%'.format(cost * 100))
        if cur_pred != true_class:
            print('Predictor is already being fooled!')
            print('New predicted class: ' + str(cur_pred) + ' with confidence of ' + str(np.amax(_pred) * 100))
            break
        if i % cost_cp_freq == cost_cp_freq - 1:
            # print(str(abs(last_cost_cp - cost)))
            if abs(last_cost_cp - cost) < cost_dif_threshold:
                print('Image seem not to be improving anymore, early termination')
                break
            last_cost_cp = cost
        i += 1
        # print(str(model.predict(hacked_image)[0]))

    # De-scale the image's pixels from [-1, 1] back to the [0, 255] range
    imgh = hacked_image[0]
    imgh *= 255.
    # print(imgh.shape)
    # print(imgh)
    # compare to the original
    dif_img = original_image - imgh

    _min = np.amin(dif_img)
    _max = np.amax(dif_img)

    # do we want the scaled noise?
    if args.noises:
        print('Scaled noise image save as ' + args.noises)
        _min = np.amin(dif_img)
        _max = np.amax(dif_img)

        scaled_noise = np.array(list(map(lambda x: ((x -_min) / abs(_max  - _min)) * 255, dif_img)))

        # Save the scaled up noise
        im = Image.fromarray(scaled_noise.astype(np.uint8))
        im.save(args.noises)

    pert = np.sum(np.absolute(dif_img))
    # print()
    # print(str(dif_img))
    # imgh = np.round(imgh)
    print('Total perturbation: ' + str(pert/255))

    # print(np.amax(imgh))
    # Save the hacked image!
    im = Image.fromarray(imgh.astype(np.uint8))
    im.save(args.save_as)
    print('Hacked image saved as ' + args.save_as)

    if args.noise:
        im = Image.fromarray(dif_img.astype(np.uint8))
        im.save(args.noise)
        print('Noise image saved as ' + args.noise)

    # save the model
    model_json = model.to_json()
    with open("DNN.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("DNN_weights.h5")
    print("Saved model to disk")

if __name__=='__main__':
    __main__()