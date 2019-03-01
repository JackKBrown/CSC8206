import numpy as np
import random
from PIL import Image

def randomNoise(img, maxRGBVal=255, p=0.1, am=255, x1=-1, x2=-1, y1=-1, y2=-1, loop=False):
    img.show()
    pix = img.load()
    copy = img.copy()
    original = copy.load()
    pixelsChanged = 0
    
    if(x1 < 0):
        x1 = 0
    if(x2 < 0):
        x2 = img.size[0]
    if(y1 < 0):
        y1 = 0
    if(y2 < 0):
        y2 = img.size[1]
    
    for i in range(y1,y2):
        for j in range(x1,x2):
            pixel = pix[i,j]
            rgbVals = list(pixel)
            if(random.uniform(0,1)<=p):
                pixelsChanged += 1
                for k in range(len(rgbVals)):                
                    noise = random.randint(1,am)   
                    r = random.randint(0,1)
                    if(r == 0):                        
                        newVal = rgbVals[k] + noise
                        if(newVal<=maxRGBVal):
                                rgbVals[k] = newVal
                        else:
                            if(loop):
                                rgbVals[k] = newVal % (maxRGBVal+1)
                            else:
                                rgbVals[k] = maxRGBVal
                    else:
                        newVal = rgbVals[k]-noise
                        if(newVal>=0):
                            rgbVals[k] = newVal
                        else:
                            if(loop):
                                rgbVals[k] = newVal % (maxRGBVal+1)
                            else:
                                rgbVals[k] = 0
            
            pix[i,j] = tuple(rgbVals)
            
    img.show()
    
    totalNoise = 0
    
    for i in range(y1,y2):
        for j in range(x1,x2):
            pixelNoise = pix[i,j]
            pixelOriginal = original[i,j]
            
            rgbValsNoise = list(pixelNoise)
            rgbValsOriginal = list(pixelOriginal)
            
            for k in range(len(rgbValsNoise)):
                totalNoise += abs(rgbValsNoise[k]-rgbValsOriginal[k])
      
    #Print total noise that was added/subtracted
    print ("Total Noise: ", totalNoise)
    print ("Number of Pixels changed: ", pixelsChanged)


img = Image.open('images_cropped/0.ppm')

#Examples
randomNoise(img, p=1, am=100) 