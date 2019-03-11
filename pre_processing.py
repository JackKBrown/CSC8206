from PIL import Image
# import cv2 as cv
import scipy.misc as smp
import csv
import os

TARGET_RESOLUTION = (40, 40)

FOLDERS = 43 # num of folders
IMG_NAME_IT = 30 # iterator of the img names within folders

PATH_BASE = 'images_orig/000'
CROPPED_IMG_PATH = 'images_cropped/'
CROPPED_IMG_PATH2 = 'images_cropped2/'
CLASS_CSV_PATH = 'classes.csv'

gl_it = 0
gl_dics = {}

# folder iteration
for i in range(FOLDERS):
    _it = ('0' if i < 10 else '') + str(i)
    path = PATH_BASE + _it + '/'
    print('Working on ' + path)

    csv_path = path + 'GT-000' + _it + '.csv'

    # ensure there is a dir for the class in cropped imgs dir
    if not os.path.exists(CROPPED_IMG_PATH + _it):
        os.makedirs(CROPPED_IMG_PATH + '000' + _it)
    if not os.path.exists(CROPPED_IMG_PATH2 + _it):
        os.makedirs(CROPPED_IMG_PATH2 + '000' + _it)

    with open(csv_path, newline='') as csvfile:
        data = list(csv.reader(csvfile))

    # files within folder iteration
    for j in range(len(data)-1):
        line_parts = data[j+1][0].split(';')
        crop_area = list(map(int, line_parts[3:7]))

        gl_dics[gl_it] = line_parts[7] # map the classId

        img_path = path + line_parts[0]
        img_orig = Image.open(img_path)
        # img_orig = cv.imread(img_path, 1)
        img_new = img_orig.crop(crop_area).resize(TARGET_RESOLUTION)# have a look into PIL.Image.LANCZOS ?

        # save with a target resolution

        #img_new.save(CROPPED_IMG_PATH + str(gl_it) + '.ppm') # use for aggregated dir save
        #img_new.save(img_path.replace('.ppm', '_cropped.ppm'))
        img_new.save(CROPPED_IMG_PATH2 + '000' + str(_it) + '/' + line_parts[0].replace('.ppm', '.png'))
        img_new.save(CROPPED_IMG_PATH + '000' + str(_it) + '/' + line_parts[0])
        # cv.imwrite(CROPPED_IMG_PATH + '000' + str(_it) + '/' + line_parts[0].replace('.ppm', '.png'), img_new)
        gl_it += 1

# saves the [id, class] pairs as csv
with open(CLASS_CSV_PATH, 'w') as f:
    for key in gl_dics.keys():
        f.write("%s,%s\n"%(key,gl_dics[key]))
