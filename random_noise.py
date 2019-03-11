#Imports
import numpy as np
import random
from PIL import Image
import demo as dnn

#Function to apply random noise to a given image, as per a set of parameters. Returns the final image
#Namely: p, probability a pixel has noise applied. This number should be between 0 and 1;
#        am, the maximum amount of noise. The noise is applied with a uniform distribution between 1
#        and am to the pixels rgb values independantly;
#        x1,x2,y1,y2, values to allow the user to target an area of the image. A negative number
#        for these values specifies that the entire image should be targeted;
#        loop, either true or false to whether the pixel rgb values should loop beyond bounds
#        if false the values will cap at the max and min rgb values
def randomNoise(img, p=0.1, am=255, x1=-1, x2=-1, y1=-1, y2=-1, loop=False, maxRGBVal=255):
    #Copy and load the original image, copy to preserve original image
    imgCopy = img.copy()
    pix = imgCopy.load()
    
    #If the x1,x2,y1,y2 values are negative, the user intends to target the whole image
    #Therefore set these values as the images bounds
    if(x1 < 0):
        x1 = 0
    if(x2 < 0):
        x2 = img.size[0]
    if(y1 < 0):
        y1 = 0
    if(y2 < 0):
        y2 = img.size[1]
    
    #For each pixel in the images specified area
    for i in range(y1,y2):
        for j in range(x1,x2):
            #Get the current pixels rgb values
            pixel = pix[i,j]
            rgbVals = list(pixel)
            #Randomly generate a number between 0 and 1, if less than p, then apply noise to the pixel
            if(random.uniform(0,1)<=p):
                #For each rgb value independantly 
                for k in range(len(rgbVals)):     
                    #Generate the amount of noise
                    noise = random.randint(1,am) 
                    #Generate either 0 or 1, to split whether noise is added or subtracted
                    r = random.randint(0,1)
                    #0 - Add noise
                    if(r == 0):
                        #Calculate the pixels final value
                        newVal = rgbVals[k] + noise
                        if(newVal<=maxRGBVal):
                                rgbVals[k] = newVal
                        else:
                            if(loop):
                                rgbVals[k] = newVal % (maxRGBVal+1)
                            else:
                                rgbVals[k] = maxRGBVal
                    #1 - Subtract Noise
                    else:
                        #Calculate the pixels final value
                        newVal = rgbVals[k]-noise
                        if(newVal>=0):
                            rgbVals[k] = newVal
                        else:
                            if(loop):
                                rgbVals[k] = newVal % (maxRGBVal+1)
                            else:
                                rgbVals[k] = 0
            
            #Update the pixel as per the randomly generated pixels
            pix[i,j] = tuple(rgbVals)

    #Return the random noise image
    return imgCopy

#Function to calculate noise between two provided images. Noise is calculate as the total absolute
#difference between the two images pixel RGB values. Returns the total calculated noise
def calculateNoise(origPath, noisePath):
    #Open and load the images
    origImg = Image.open(origPath)
    noiseImg = Image.open(noisePath)    
    orig = origImg.load()
    noise = noiseImg.load()
    
    #Running counter for total noise
    totalNoise = 0
    
    #Check the images are the same dimensions
    if(origImg.size[1] != noiseImg.size[1]) or (origImg.size[0] != noiseImg.size[0]):
        print("The images at the provided paths are different sizes.")
    else:
        #For each pixel, get the rgb values of both images
        for i in range(origImg.size[0]):
            for j in range(origImg.size[1]):
                origPixel = orig[i,j]
                noisePixel = noise[i,j]
                origRGBVals = list(origPixel)
                noiseRGBVals = list(noisePixel)
                
                #For each RGB value, add the difference between both images to the total noise counter
                for k in range(len(origRGBVals)):
                    totalNoise += abs(origRGBVals[k] - noiseRGBVals[k])
    #Return the total noise
    return totalNoise

#A function to calculate the minimal perturbation of an image, to fool the dnn model.
def minimisePerturbation(origPath, savePath, iterationsBeforeGiveUp, startP=1, pStep=0.05, startAm=255, amStep=5, x1=-1,x2=-1,y1=-1,y2=-1):
    #Load the model, original image, and original image class
    #The function expects that the original image correctly classifies
    model = dnn.load_DNN()
    origImg = Image.open(origPath)
    origClass = dnn.test_image(origPath, model).argmax()

    #Set current p and am as the start p and am respectively
    currentP = startP
    currentAm = startAm

    #Initialise current min noise as a number larger than possible from any perturbation, such that any result performs better
    currentMinNoise = 9999999999999
    #Variables to store the currently found minimum p and am, -1 shows no result yet found
    currentMinP = -1
    currentMinAm = -1

    #Booleans to control the flow of the function
    resultFoundP = True
    resultFoundAm = True

    #If true, a result was found at the previous p, first iteration is also true
    #If no result found at the previous p, then exit the calculations and return minimum found
    while(resultFoundP):
        #Set false, therefore if no result found, it will remain false and the loop will exit after this iteration
        resultFoundP = False

        #If true, a result was found at the previous am value
        #If no result found at previous am, then it will not try lower am values
        while(resultFoundAm):
            #Set false, therefore if no result found, it will remain false and the loop will exit after this iteration
            resultFoundAm = False
            
            #i, which iteration we are on for this p and am amount
            i = 0

            print("Testing P: " , currentP, " Am: " , currentAm)

            #If the number of iterations is less than the specified giveup amount
            while(i < iterationsBeforeGiveUp):
                #Generate and save a perturbated image with the provided parameters
                imgPer = randomNoise(origImg, p=currentP, am=currentAm, x1=x1, x2=x2, y1=y1, y2=y2)
                imgPer.save(savePath + '/tmp.ppm')
                #If this image misclassifies in dnn, it's a solution
                if(dnn.test_image((savePath + '/tmp.ppm'), model).argmax() != origClass):
                    print("Solution Found")
                    #Update the 'flow' booleans to allow the next step down to be tested
                    resultFoundAm = True
                    resultFoundP = True

                    #Calculate the noise of this image
                    imgPerNoise = calculateNoise(origPath, savePath + '/tmp.ppm')
                    #Check if a new minimum solution, update if so
                    if(imgPerNoise < currentMinNoise):
                        print("Solution is current min with Total Noise: " , imgPerNoise)
                        imgPer.save(savePath + '/currentMin.ppm')
                        currentMinNoise = imgPerNoise
                        currentMinP = currentP
                        currentMinAm = currentAm
                    #Break out of this p, am test, since a result has been found. No need to test the 
                    #rest of iteration before giveup, since will produce almost identical results.
                    break;
                #No result found, increment i and try again
                i+=1
           
            currentAm -= amStep
        
        currentP -= pStep
        #Round to prevent slight float difference (e.g 0.799999999999 becomes 0.8)
        currentP = round(currentP, 8)
        #Reset am ready for next p value testing
        currentAm = startAm
        resultFoundAm = True

#The code to run the above defined functions
imgPath = 'images_cropped/00000/00000_00000.ppm'
savePath = 'D:/Users/eddie/Desktop'

minimisePerturbation(imgPath, savePath, 50)
    