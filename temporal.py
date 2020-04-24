'''
This script analyze an specific region in coordinatesPath, of a set of images located at path, doing the following 
methodology:

1 - Segmenting possible patches of birds in the image
2 - Classifying each patch with a CNN model trained on tensorflow
3 - Counting the amount of birds in each image and storing the values in an array

The algotihm have a low computational cost, and was integrated in PsophiaTool software. 
'''


import cv2
import numpy as np
import funciones
from tensorflow.keras.models import load_model
import os
import sys
from getpass import getuser
from skimage.measure import label, regionprops, regionprops_table
import tensorflow as tf
import xlsxwriter 


# Import Models
new_model = tf.keras.models.load_model('./Models/MobileNetV18.h5', compile=False)

#Read User
username = getuser()

#Read coordinates
coordinatesPath =  'C:/Users/' + username +'/AppData/Local/DatosPsophiaTool/coordinatesCropping.txt'
coordinatesStr = funciones.LeerCoordenadas(coordinatesPath)

#Images files path
path = 'C:/Users/' + username +'/AppData/Local/DatosPsophiaTool/Ruta.txt' 

#Variables 
tendency = [] # array of [sum(:), frame]
values = np.array([])
names = [] 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))

#OpenPath
f = open(path, "r") 
lt = 1
mu = 1

#Itetarions over the each images
for x in f:
    #Step 0: Read the path
    x = x.replace("\n", "") 
    path = x.replace("\n", "") 
    BASE = path.split('/')[-2]
    dirname = path.split('/')[-1]
    nombreGuardar = x.split('/')[-2]+'_'+x.split('/')[-1] # With this, we can have tha label frameXX.jpg


    #Step 1: Read the image from the absolute path 
    im = cv2.imread(str(x), 0)
    im2 = cv2.imread(str(x))
    
    #Step 1.1: Original sector analysis
    imO = im
    imO = imO[coordinatesStr[0]:coordinatesStr[2], coordinatesStr[1]:coordinatesStr[3]]
    im2 = im2[coordinatesStr[0]:coordinatesStr[2], coordinatesStr[1]:coordinatesStr[3],:]
    im2 = funciones.NormalizacionIm2(im2) # Normalizing RGB image into multiple of 256
    imO = funciones.NormalizacionIm(imO) # Normalizing GrayScale image into multiples o 256
    fil, col = imO.shape

    #Step 1.2: Image Preprocessing
    img = imO
    img = cv2.medianBlur(img, 5)
    img = cv2.GaussianBlur(img,(5,5),0.6,sigmaY=0.6)

    #Step 1.3: Image Segmentation
    imT = funciones.Segmentado(img, kernel)

    #Step 1.4: Regionprops
    label_img = label(imT)
    regions = regionprops(label_img)
    #Filter by area
    X = funciones.filtroAreas(regions)
    Xtest1, Xtest, New_Coordinates = funciones.ROI_Analysis(im2, X, fil, col) #ROI proposals
    
    #Step 2: Classifying local patches 
    Bandera = 1 # Local variable that allows to see the patches prediction with it's probability
    if(len(New_Coordinates["X1"]) >  0 and Xtest.shape[0] > 1):
            score = funciones.predecir2(Xtest1, new_model)
            New_Coordinates, scoreC = funciones.Survive(New_Coordinates,score)
            
            if (Bandera == 1):
                img2 = funciones.Mostrar_Boxes(im2, New_Coordinates, scoreC)
                cv2.imwrite('./Resultados/' + dirname + '.JPG', img2)
            
    # Displaying a message for the user
    message = mu * ' * '
    if (mu > 10):
        mu = 1
    mu = mu + 1
    print(message)
    sys.stdout.flush()

    #The length of the New_Coordinates["X1"] give us the total amount of birds detected
    if(len(New_Coordinates["X1"]) >  0):
    	valor = len(New_Coordinates["X1"])
    else:
    	valor = 0

    values = np.append(values,valor)
    arreglo = x.split('/')[-1] # With this, we can have tha label frameXX.jpg
    names.append(dirname)


#Organizing the information into a list
for i in range(len(values)): 
    tendency.append([names[i],values[i]])


pathSaveCSV = './' + BASE + '.xlsx'
workbook = xlsxwriter.Workbook(pathSaveCSV)
worksheet = workbook.add_worksheet("Analisis")  

# Start from the first cell. Rows and 
# columns are zero indexed. 
row = 0
col = 0

# Iterate over the data and write it out row by row. 
for name, score in (tendency): 
      worksheet.write(row, col, name) 
      worksheet.write(row, col + 1, score) 
      row += 1
workbook.close() 


print('The segmentation was sucessfully done')
sys.stdout.flush()