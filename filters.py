from PIL import Image

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
	
def mask(image_path, mask_path, threshold):

	# get original image
	im = Image.open(image_path)
	pix = im.load()
	
	# get image mask
	mask = Image.open(mask_path)
	mask_pix = mask.load()
	
	# create masked image
	width, height = im.size
	for x in range(width):
		for y in range(height):
			if (mask_pix[x,y] > (threshold,0,0)): # lower threshold = more mask
				pix[x,y] = (0,0,0)
	
	im.show()
	
################	
#modify_val('images_cropped/10859.ppm', -100)
mask('images_cropped/10859.ppm', 'images_noise/circle_mask.png', 220)