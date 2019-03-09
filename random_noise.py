import numpy as np
import random
from PIL import Image
import demo as dnn

def randomNoise(img, maxRGBVal=255, p=0.1, am=255, x1=-1, x2=-1, y1=-1, y2=-1, loop=False):
    imgCopy = img.copy()
    pix = imgCopy.load()
    
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

    return imgCopy

def compareNoise(origPath, noisePath):
    origImg = Image.open(origPath)
    noiseImg = Image.open(noisePath)
    
    orig = origImg.load()
    noise = noiseImg.load()
    
    totalNoise = 0
    
    if(origImg.size[1] != noiseImg.size[1]) or (origImg.size[0] != noiseImg.size[0]):
        print("The images at the provided paths are different sizes.")
    else:
        for i in range(origImg.size[0]):
            for j in range(origImg.size[1]):
                origPixel = orig[i,j]
                noisePixel = noise[i,j]
                origRGBVals = list(origPixel)
                noiseRGBVals = list(noisePixel)

                for k in range(len(origRGBVals)):
                    totalNoise += abs(origRGBVals[k] - noiseRGBVals[k])

    return totalNoise


def minimisePerturbation(origPath, savePath, iterationsBeforeGiveUp, startP=1, pStep=0.05, startAm=255, amStep=5, x1=-1,x2=-1,y1=-1,y2=-1):
    model = dnn.load_DNN()
    origImg = Image.open(origPath)
    origClass = dnn.test_image(origPath, model).argmax()

    currentP = startP
    currentAm = startAm

    #Initialise current min noise as a number larger than possible from any perturbation, such that any result performs better
    currentMinNoise = 9999999999999
    currentMinP = -1
    currentMinAm = -1

    resultFoundP = True
    resultFoundAm = True

    while(resultFoundP):
        resultFoundP = False

        while(resultFoundAm):
            resultFoundAm = False
            i = 0

            print("Testing P: " , currentP, " Am: " , currentAm)

            while(i < iterationsBeforeGiveUp):
                imgPer = randomNoise(origImg, p=currentP, am=currentAm, x1=x1, x2=x2, y1=y1, y2=y2)
                imgPer.save(savePath + '/tmp.ppm')
                if(dnn.test_image((savePath + '/tmp.ppm'), model).argmax() != origClass):
                    print("Solution Found")
                    resultFoundAm = True
                    resultFoundP = True

                    imgPerNoise = compareNoise(origPath, savePath + '/tmp.ppm')
                    if(imgPerNoise < currentMinNoise):
                        print("Solution is current min with Total Noise: " , imgPerNoise)
                        imgPer.save(savePath + '/currentMin.ppm')
                        currentMinNoise = imgPerNoise
                        currentMinP = currentP
                        currentMinAm = currentAm

                    break;
                i+=1
            currentAm -= amStep
        
        currentP -= pStep
        currentP = round(currentP, 8)
        currentAm = startAm
        resultFoundAm = True

imgPath = 'images_cropped/00000/00000_00000.ppm'
savePath = 'D:/Users/eddie/Desktop'

minimisePerturbation(imgPath, savePath, 50)
    