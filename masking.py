from PIL import Image
from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
import demo as dnn
import shutil

def modify_val(image_path, value):

	# get original image
	im = Image.open(image_path) 
	pix = im.load()

	width, height = im.size

	for x in range(width):
		for y in range(height):
			pixel = pix[x,y]
			rgb_vals = list(pixel)
			for i in range(len(rgb_vals)):
				rgb_vals[i] += value
			
			pix[x,y] = tuple(rgb_vals)
			
			
	im.show()
	
def mask(image_path, mask_path, mask_translation, threshold, min_perturbation):

	# get original image
	im = Image.open(image_path)
	pix = im.load()
	
	# to calculate the difference later
	old_pix = Image.open(image_path).load()
	
	# get translated image mask
	mask_pix = translate_img(mask_path, mask_translation[0], mask_translation[1])
	
	# create masked image
	width, height = im.size
	for x in range(width):
		for y in range(height):
			if (mask_pix[x,y] > (threshold,0,0)): # lower threshold = more mask
				pix[x,y] = (0,0,0)
	
	# calculate difference between original and final picture
	total_diff = 0
	for x in range(width):
		for y in range(height):
			rd = abs(old_pix[x,y][0] - pix[x,y][0])
			gd = abs(old_pix[x,y][1] - pix[x,y][1])
			bd = abs(old_pix[x,y][2] - pix[x,y][2])
			
			total_diff += rd + gd + bd
	
	if total_diff < min_perturbation:
		im.save("masked_opaque.ppm")
		
	return total_diff
	
# unused as the effort switched to the opaque masking method
def mask_blend(image_path, mask_path, mask_translation, mask_transparency):
	# get original image
	im = Image.open(image_path)
	pix = im.load()
	
	# to calculate the difference later
	old_pix = Image.open(image_path).load()
	
	# get image mask
	mask_pix = translate_img(mask_path, mask_translation[0], mask_translation[1])
	
	width, height = im.size
	total_diff = 0
	for x in range(width):
		for y in range(height):
			a_mask = (mask_transparency / 255) * (mask_pix[x,y][3] / 255)
			r = mask_pix[x,y][0] * a_mask + pix[x,y][0] * (1 - a_mask)
			g = mask_pix[x,y][1] * a_mask + pix[x,y][1] * (1 - a_mask)
			b = mask_pix[x,y][2] * a_mask + pix[x,y][2] * (1 - a_mask)
			
			total_diff += sqrt(pow(pix[x,y][0] - r,2)+pow(pix[x,y][1] - g,2)+pow(pix[x,y][2] - b,2))
			pix[x,y] = (int(r),int(g),int(b))
				
	max_diff = 706676.7294881183 # white to black image
	diff_percentage = total_diff * 100.0 / max_diff
	#print("Total difference: ", diff_percentage, "%\t(absolute value:", total_diff, ")")
	
	im.show()
	im.save("masked_transparency.ppm")
	
	
def translate_img(image_path, x_translation, y_translation):
	# to do: fix horizontal wrapping

	# load image
	im = Image.open(image_path)
	pix = im.load()
	width, height = im.size
	
	# translation value in pixels
	total_translation = y_translation * width + x_translation
	
	# independent copy of pixel values
	old_pix = Image.open(image_path).load()
	
	for x in range(width):
		for y in range(height):
			current_pixel = y * width + x
			old_pixel = current_pixel - total_translation
			
			if (old_pixel >=0 and old_pixel < width * height): # so long as the old pixel is valid
				pix[x,y] = old_pix[old_pixel % width, old_pixel / width]
			else:
				pix[x,y] = (0,0,0)
	
	return pix


def diff_images(path_orig, path_mod):
	im_orig = Image.open(path_orig)
	pix_orig = im_orig.load()
	pix_mod = Image.open(path_mod).load()
	width, height = im_orig.size
	
	total_difference = 0
	
	for x in range(width):
		for y in range(height):
			total_difference = total_difference + abs(pix_orig[x,y][0] - pix_mod[x,y][0]) + abs(pix_orig[x,y][1] - pix_mod[x,y][1]) + abs(pix_orig[x,y][2] - pix_mod[x,y][2])
	
	
	return total_difference

def mask_compute(img_orig, img_mask, out_path, model):
	class_type = int(dnn.test_image(img_orig, model).argmax())
	
	min_perturbation = 999999999999999999
	
	# for optimisation, as a central pixel will likely require less noise than a corner one for misclassification
	for i in reversed(range(255)):
		pert = mask(img_orig, img_mask, (0,0), i, min_perturbation)
		if pert >= min_perturbation:
			break
		if int(dnn.test_image('masked_opaque.ppm', model).argmax()) is not class_type:
			min_perturbation = pert
			shutil.copyfile('masked_opaque.ppm', out_path)
			break

	for x in range(-20, 21):
		for y in range(-20, 21):
			print((x,y), "\t Current minimum perturbation: ", min_perturbation)
			for i in reversed(range(255)):
				pert = mask(img_orig, img_mask, (x, y), i, min_perturbation)
				if pert >= min_perturbation:
					break
				if int(dnn.test_image('masked_opaque.ppm', model).argmax()) is not class_type:
					min_perturbation = pert
					shutil.copyfile('masked_opaque.ppm', out_path)
					break
	
	return min_perturbation

def compare_images(img_orig, img_pert, model):
	orig_class = dnn.test_image(img_orig, model).argmax()
	pert_class = dnn.test_image(img_pert, model).argmax()
	
	diff = diff_images(img_orig, img_pert)
	
	print("Original class: ", orig_class, "\tModified class: ", pert_class, "\tDifference: ", diff)
	return orig_class, pert_class
	
################

model = dnn.load_DNN()
mask_path = 'images_noise/circle_mask.png'
img_orig = ('images_demo/00000/00001_00006.ppm', 'images_demo/00004/00034_00015.ppm', 'images_demo/00008/00004_00010.ppm', 'images_demo/00023/00013_00009.ppm', 'images_demo/00032/00000_00003.ppm')
out_img = 'masked_min.ppm'

mask_compute(img_orig[0], mask_path, out_img, model)
compare_images(img_orig[0], out_img, model)
