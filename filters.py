from PIL import Image

def increase_val():
	im = Image.open('images_cropped/10859.ppm') 
	pix = im.load()

	w, h = im.size

	for x in range(w):
		for y in range(h):
			t = pix[x,y]
			l = list(t)
			for i in range(len(l)):
				l[i] += 100
			
			pix[x,y] = tuple(l)
			
			
	print(pix[10,20])
	im.show()  
	

def noise_mask():
	im = Image.open('images_cropped/10859.ppm')
	noise = Image.open('images_noise/white_noise.png')
	pix = im.load()
	npix = noise.load()

	w, h = im.size
	for x in range(w):
		for y in range(h):
			if (npix[x,y] < (100,0,0)):
				pix[x,y] = (0,0,0)
			
	im.show()
	
noise_mask()