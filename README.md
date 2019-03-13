# CSC8206
Repository for Newcastle university group 9 project in security and resilience. 

contributors:
Jack K. Brown,  Giovanni Solaini,  Vaclav Vydra,  Edward C. Jacobs,  Laheem Khan

This project aims to find the minimum perturbation on an image to force a Deep Neural Network classifier to misclassify that image.
We have used a simple DNN for this and are treating it as a black box to find techniques for finding the minimum perturbation.

contains:
DNN.py - the deep neural network model

demo.py - module for running the DNN and when run contains a demo for image 00000/00000_0000.ppm

images_* - these folders are full of the images used to train the model these are sourced from the institute for neuroinformatics http://benchmark.ini.rub.de

random_noise.py - file containing the functions to define and run the random_noise algorithm. See code comments for more details
				- to run random_noise algorithm, first change the file paths at the bottom of the code to work for your example, then jrun to generate min perturbation for your example
				
This repository is for educational purposes.
