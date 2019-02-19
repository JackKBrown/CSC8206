from PIL import Image
import scipy.misc as smp

im = Image.open('images_cropped/10859.ppm') # open the image

pixels = list(im.getdata()) # extracts the pixels
width, height = im.size
pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]

img = smp.toimage(pixels)       # Create a PIL image
img.show()                      # View in default viewer