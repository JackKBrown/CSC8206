import numpy as np
import matplotlib.pyplot as plt
import random

#Initialise an array of size 40x40, with all elements being 120
#This represents an image of 40px by 40px in a mid-grey colour (assuming 0-255 value range)
grey = np.zeros(shape=(40,40))
for i in range(40):
    for j in range(40):
        grey[i,j] = 120
		

#Function to create random noise for a given image (np array), with parameters for probability of noise
#for a specific pixel, p, and an amount of noise, am. The amount represents the upperbound of possible noise applied,
#for each pixel, being randomly chosen from 1 to amount if a pixel has noise applied. The function will also display
#the total amount of noise applied
def randomNoise(img, p=1, am=255):
    #Copy image to new array so noise can be applied
    imgNoise = np.copy(img)
    
    #Running total for noise amount
    totalNoise = 0
    #Min and max pixel values
    maxPixVal = 255
    minPixVal = 0

    #For each pixel in the array
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #Generate random float from 0 to 1
            r = random.uniform(0,1)
            #If this generated float is less than p, apply noise to this pixel
            if(r<=p):
                #Generate amount of noise to be applied
                noise = random.randint(1,am+1)
                #Random number of 0 or 1, 0 represents that noise will be added, 1 represents noise will be subtracted
                r = random.randint(0,1)
                #Add noise
                if(r == 0):
                    #Value of pixel after noise applied
                    addNoise = img[i,j]+noise
                    #If valid pixel value
                    if(addNoise<=maxPixVal):
                        #Set the new value in imgNoise
                        imgNoise[i,j] = addNoise
                        #Add to total noise the value of noise added
                        totalNoise = totalNoise + noise
                    #Else, the value is above the pixel value range
                    else:
                        #Set as max value
                        imgNoise[i,j] = maxPixVal
                        #Add the effective noise that has been added
                        totalNoise = totalNoise + (maxPixVal - img[i,j])                        
                #Subtract noise
                else:
                    #Calculate noise value after subtraction
                    subNoise = img[i,j]-noise
                    #If valid pixel value
                    if(subNoise>=minPixVal):
                        #Set new value in imgNoise
                        imgNoise[i,j] = subNoise
                        #Add the noise that was subtracted to the running total
                        totalNoise = totalNoise + noise
                    #Else, the value is below the pixel value range
                    else:
                        #Set as min pixel value
                        imgNoise[i,j] = minPixVal
                        #Add the effective noise that was subtracted from original value
                        totalNoise = totalNoise + (img[i,j]) 
    
    #Below plots and shows the original, and noise adjusted images.
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("Original Image")
    plt.show()
    
    plt.gray()
    plt.imshow(imgNoise, cmap='gray', vmin=0, vmax=255)
    plt.title("Noise Image")
    plt.show()
    
    #Print total noise that was added/subtracted
    print("Total Noise: ", totalNoise)


randomNoise(grey,0.15,25) 