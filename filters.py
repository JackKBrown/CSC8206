from PIL import Image
from keras.preprocessing import image
from keras.models import model_from_json
from math import sqrt
import numpy as np

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
	
def mask(image_path, mask_path, mask_translation, threshold):

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
			rd = old_pix[x,y][0] - pix[x,y][0]
			gd = old_pix[x,y][1] - pix[x,y][1]
			bd = old_pix[x,y][2] - pix[x,y][2]
			
			total_diff += sqrt(pow(rd,2)+pow(gd,2)+pow(bd,2))
			
		
	max_diff = 706676.7294881183 # white to black image
	diff_percentage = total_diff * 100.0 / max_diff
	print("Total difference: ", diff_percentage, "%\t(absolute value:", total_diff, ")")
	#im.show()
	im.save("masked_opaque.ppm")
	
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
	print("Total difference: ", diff_percentage, "%\t(absolute value:", total_diff, ")")
	
	#im.show()
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


################	
#modify_val('images_cropped/10859.ppm', -100)
mask_blend('images_cropped/00000/00000_00011.ppm', 'images_noise/circle_mask.png', (10, 0), 150)
def extract_noise_mask(orig_image_path, hacked_image_path):
	im1 = image.load_img(orig_image_path)  # open the image
	pixels1 = image.img_to_array(im1)  # extracts the pixels

	im2 = image.load_img(hacked_image_path)  # open the image
	pixels2 = image.img_to_array(im2)  # extracts the pixels

	noise = []
	for p1, p2 in zip(pixels1, pixels2):
		noise += p1 - p2

def predict(img_path):
	im = image.load_img(img_path, target_size=(40,40)) # open the image
	pixels = image.img_to_array(im) # extracts the pixels
	pixels = [x/.255 for x in pixels]
	pixels = np.reshape(pixels, newshape=(40, 40, 3))
	pixels = np.expand_dims(pixels, axis=0)
	
	# load json and create model
	json_file = open('DNN.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights("DNN_weights.h5")
	#print("Loaded model from disk")
	pred = model.predict(pixels)

	print('Class id: ', pred[0][1])
	return int(pred[0][1])
################	
#modify_val('images_cropped/10859.ppm', -100)

class_type = predict('images_orig/00000/00000_00000.ppm')


for i in reversed(range(255)):
	mask('images_cropped/00000/00000_00000.ppm', 'images_noise/circle_mask.png', (0, 0), i)
	if predict('masked_opaque.ppm') is not class_type:
		Image.open('masked_opaque.ppm').show()
		break
