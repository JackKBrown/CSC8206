from PIL import Image
from PIL import PpmImagePlugin

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
	
	# get translated image mask
	mask_pix = translate_img(mask_path, mask_translation[0], mask_translation[1])
	
	# create masked image
	width, height = im.size
	for x in range(width):
		for y in range(height):
			if (mask_pix[x,y] > (threshold,0,0)): # lower threshold = more mask
				pix[x,y] = (0,0,0)
	
	im.show()
	
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
				
	im.show()
	
	return pix

################	
#modify_val('images_cropped/10859.ppm', -100)
mask('images_cropped/10859.ppm', 'images_noise/circle_mask.png', (10, 10), 220)