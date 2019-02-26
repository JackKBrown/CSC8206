import numpy as np
import matplotlib.pyplot as plt
import random
#from PIL import Image

#Initialise an array of size 40x40, with all elements being 120
#This represents an image of 40px by 40px in a mid-grey colour (assuming 0-255 value range)
grey = np.zeros(shape=(40,40))
for i in range(40):
    for j in range(40):
        grey[i,j] = 120
		
#Function to create random noise for a given image, with various parameters described below
#min/maxPixVal, minimum and maximum value for each pixel
#p, probability a pixel will have noise applied. am, the max amount of noise, will generate uniformly from 1 to am
#x1,x2,y1,y2, allows specification of area of image for noise to be applied, a number <0 will do the bounds of the image
#loop, specifies if you'd like noise to loop or max/min out at value bounds
#The function will also display the total amount of noise applied
def randomNoise(img, maxPixVal=255, minPixVal=0, p=0.1, am=255, x1=-1, x2=-1, y1=-1, y2=-1, loop=True):
    #Copy image to new array to allow noise to be added
    imgNoise = np.copy(img)
    
    #If x1,x2,y1,y2 not specified, set as 0 and image size respectively (to apply noise to whole image)
    if(x1 < 0):
        x1 = 0
    if(x2 < 0):
        x2 = img.shape[0]
    if(y1 < 0):
        y1 = 0
    if(y2 < 0):
        y2 = img.shape[1] 
    
    #For each pixel in specified area
    for i in range(y1,y2):
        for j in range(x1,x2):
            #Generate random float from 0 to 1, if less than p, apply noise to the pixel
            if(random.uniform(0,1)<=p):
                #Generate amount of noise to be applied.
                noise = random.randint(1,am+1)                
                #Random number of 0 or 1, 0 represents that noise will be added, 1 represents noise will be subtracted
                r = random.randint(0,1)
                
                #Add noise
                if(r == 0):                        
                    #Value of pixel after noise applied
                    newVal = img[i,j]+noise
                    #If valid pixel value
                    if(newVal<=maxPixVal):
                        #Set the new value in imgNoise
                        imgNoise[i,j] = newVal
                    #Else, the value is above the pixel value range
                    else:
                        #If looping
                        if(loop):
                            #New noisy value is the newValue modulo maxPixVal
                            imgNoise[i,j] = newVal % maxPixVal
                        #Else, not looping
                        else:
                            #Set as max value
                            imgNoise[i,j] = maxPixVal
                        
                #Else subtract noise
                else:
                    #Calculate noise value after subtraction
                    newVal = img[i,j]-noise
                    #If valid pixel value
                    if(newVal>=minPixVal):
                        #Set new value in imgNoise
                        imgNoise[i,j] = newVal
                    #Else, the value is below the pixel value range
                    else:
                        #If looping
                        if(loop):
                            #New noisy value is the value of noise after addition
                            imgNoise[i,j] = 255 - (newVal % maxPixVal)
                        #Else, not looping
                        else:
                            #Set as min pixel value
                            imgNoise[i,j] = minPixVal

    #Running total for noise applied
    totalNoise = 0
    
    #Below calculates the final effective noise by summing absolute difference for each pixel in
    #the original image and the noisy image
    for i in range(y1,y2):
        for j in range(x1,x2):
            totalNoise += abs(img[i,j]-imgNoise[i,j])
    
    
    #Below plots and shows the original, and noise adjusted images.
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("Original Image")
    plt.show()
    
    plt.imshow(imgNoise, cmap='gray', vmin=0, vmax=255)
    plt.title("Noise Image")
    plt.show()
    
    #Print total noise that was added/subtracted
    print("Total Noise: ", totalNoise)


#Examples
randomNoise(grey, p=1, am=55555555) 

