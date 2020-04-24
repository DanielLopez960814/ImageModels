import cv2
import numpy as np
import scipy as sp
from sklearn.metrics import classification_report,accuracy_score
from skimage.measure import label, regionprops, regionprops_table
import math




def NormalizacionIm(Im):
    '''
    THIS FUNCTION NORMALIZE THE ROWS AND COLUMBS OF A GRAYSCALE IMAGE INTO MULTIPLES OF 256
    '''
    fil, col = Im.shape 
    PlusFil = fil % 256 
    PlusCol = col % 256 
    fil = fil + PlusFil
    col = col + PlusCol
    Im = cv2.resize(Im, (col, fil)) 
    return Im

def NormalizacionIm2(Im):
    '''
    THIS FUNCTION NORMALIZE THE ROWS AND COLUMBS OF A RGB IMAGE INTO MULTIPLES OF 256
    '''
    fil, col, _ = Im.shape #fil = filas, col = columnas
    PlusFil = fil % 256 
    PlusCol = col % 256 
    fil = fil + PlusFil
    col = col + PlusCol
    Im = cv2.resize(Im, (col, fil))
    return Im

def Survive(X, score):
    '''
    THIS FUNCTION FILTERS THE BOUNDING BOX SPECIFIED BY X[X1], X[X2], X[Y1], X[Y2]
    THAT have a score greater than 0.5 and return the scores of the filters boxes and the boxes
    '''
    X1, Y1, X2, Y2 = X["X1"], X["Y1"], X["X2"], X["Y2"]
    Y = {}
    Y["X1"] = []
    Y["Y1"] = []
    Y["X2"] = []
    Y["Y2"] = []
    scoreC = []
    for i in range(score.shape[0]):
        valor = score[i][0]
        if(valor > 0.5):
            Y["X1"].append(X1[i])
            Y["Y1"].append(Y1[i])
            Y["X2"].append(X2[i])
            Y["Y2"].append(Y2[i])
            scoreC.append(score[i][0])
    return Y, scoreC

def filtroAreas(regions):
    '''
    THIS FUNCTION FILTERS A 'regionprops sklearn object' THAT HAVE AND AREA GREATER THAN 100
    '''

    x1 = []
    x2 = []
    y1 = []
    y2 = []

    for props in regions:
        minr, minc, maxr, maxc  = props.bbox
        if props.area > 100:
            x1.append(minr)
            y1.append(minc)
            x2.append(maxr)
            y2.append(maxc)
    X = {}
    X["X1"] = x1
    X["X2"] = x2
    X["Y1"] = y1
    X["Y2"] = y2

    return X

def repartir(img):

    '''
    THIS FUNCTION GENERATES A DICTIONARY THAT CONTAINS MULTIPLES SEGMENTS OF THE GRAYSCALE IMAGE img
    '''

    a ={}
    # Parametros de la ventana
    n_H_prev = img.shape[0]
    n_W_prev = img.shape[1]
    f = 256
    pad = 0
    stride = 128
    n_H = ((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = ((n_W_prev - f + 2 * pad) / stride) + 1
    l = 0

    for h in range(int(n_H)):
        for w in range(int(n_W)):
            vert_start = h * stride
            vert_end = vert_start + f
            horiz_start = w * stride
            horiz_end = horiz_start + f
            a_slice_prev = img[vert_start:vert_end, horiz_start:horiz_end]
            a[str(l)] = a_slice_prev
            l = l + 1
    
    return a

def juntar(image, a):

    '''
    THIS FUNCTION GENERATES AND IMAGE FROM THE PATCHES SAVED IN THE DICTIONARY a
    '''
    #[fil, col] = size(image);
    fil = image.shape[0]
    col = image.shape[1]
    n_H_prev = fil
    n_W_prev = col
    f = 256
    pad = 0
    
    stride = 128
    n_H = int(((n_H_prev - f + 2 * pad) / stride) + 1)
    n_W = int(((n_W_prev - f + 2 * pad) / stride) + 1)
    im2 = 0*image
    l = 0

    #Sliding window approach
    for h in range(n_H):
        for w in range(n_W):
            vert_start = h * stride 
            vert_end = vert_start + f 
            horiz_start = w * stride 
            horiz_end = horiz_start + f 
            im2[vert_start:vert_end, horiz_start:horiz_end] = im2[vert_start:vert_end, horiz_start:horiz_end]  + a[str(l)]
            #im2[vert_start:vert_end, horiz_start:horiz_end,:] = a[str(l)]
            l = l + 1
    im2 = im2.astype(np.uint8)
    return im2


def predecir2(X,Mod_Clasificador):
    '''
    THIS FUNCTION PREDICTS THE SCORE OF A TENSOR X THAT CONTAINS THE ROI SECTORS OF THE IMAGE BASED ON A TF MODEL
    '''
    out = Mod_Clasificador.predict(X)
    
    return out


def LeerCoordenadas(coordinatesPath):
    '''
    THIS FUNCTION READ THE COORDINATES CROPPED BY THE MODULE CROP AND SAVE IN A LIST coordinatesStr
    '''
    file = open(coordinatesPath, 'r')
    coordinatesStr = file.read()

    ## Turn string [(x, y), (w,z)] into numb [x,y,w,z]
    coordinatesStr = coordinatesStr.strip('][').strip('()')
    coordinatesStr = coordinatesStr.split(',')
    coordinatesStr[1] = coordinatesStr[1].strip(')')
    coordinatesStr[2] = coordinatesStr[2].strip(' (')
    coordinatesStr = [int(i) for i in coordinatesStr]
    return coordinatesStr

def Segmentado(img, kernel):
    '''
    THIS FUNCTION SEGMENTS THE IMAGE img WITH A CV2 KERNEL kernel
    '''
    a = repartir(img)
    d = {}
    l = 0
    
    
    for key in a:	
		
        grayb = a[key]
        grayb = (grayb)/1.0
        for i in range(10):
           grayb = np.abs(grayb - np.mean(grayb))
        grayb = grayb.astype(np.uint8)
        grayb = grayb > 20 #Threshoding values greater than 20
        grayb = grayb * 255
        grayb = grayb.astype(np.uint8)
        grayb = cv2.morphologyEx(grayb, cv2.MORPH_OPEN, kernel)
        grayb = cv2.dilate(grayb, kernel, iterations=8) #iterations: hyperparameter
        d[str(l)] = grayb
        l = l + 1


    imT = juntar(img, d)
    imT = imT > 0
    return imT


def ROI_Analysis(im2, X, fil, col):
    '''
    THIS FUNCTION GENERATES A SET OF PATCHES THAT COULD BE POSSIBLY BIRDS IN THE IMAGE im2, BASED ON THE
    BOUNDING BOXES X.
    '''
    Xtest = [] # Array that will contain the patches specified by the boxes X[X1], X[X2], X[Y1], X[Y2]
    k = 0
    New_Coordinates = {} #Dictionary that will contain the filters boxes specified by X
    New_Coordinates["X1"] =  []
    New_Coordinates["X2"] =  []
    New_Coordinates["Y1"] =  []
    New_Coordinates["Y2"] =  []
    
    # For cicle to analyze the boxes X
    for i in range( len(X["X1"]) ):
        #Extracts coordinates with a banwitdh of -10/+10
        x1pp = (X["X1"][i] - 10)
        x2pp = X["X2"][i] + 10      
        y1pp = (X["Y1"][i] - 10)   
        y2pp = X["Y2"][i] + 10

        #Border condition
        if (x1pp < 0):
            x1pp = 0
        if (y1pp < 0):
            y1pp = 0
        if (x2pp >= fil):
            x2pp = fil - 1
        if (y2pp >= col):
            y2pp = col - 1
  
        ImgLocal = im2[x1pp:x2pp,y1pp:y2pp,:]
        valor1 = ImgLocal.shape[0]/ImgLocal.shape[1]

        #Filters the boxes by its shape
        if (ImgLocal.shape[0] > 66 and ImgLocal.shape[1] > 66 
        and valor1 > 0.5 and valor1 < 2):
            image = cv2.resize(ImgLocal,(128, 128), interpolation=cv2.INTER_NEAREST)
            New_Coordinates["X1"].append( x1pp )
            New_Coordinates["X2"].append( x2pp )
            New_Coordinates["Y1"].append( y1pp )
            New_Coordinates["Y2"].append( y2pp )
            Xtest.append(image.astype(float)/255.)  
            k = k + 1

    Xtest = np.array(Xtest)

    #Creating numpy array
    if(Xtest.shape[0] > 1):
	    Xtest1 = np.zeros((Xtest.shape[0],128,128,3))
	    for i in range(Xtest.shape[0]):
	    	Xtest1[i] = Xtest[i]
    
    return Xtest1, Xtest, New_Coordinates


def Mostrar_Boxes(im2, New_Coordinates, scoreC):

    '''
    THIS FUNCTION GENERATES AN IMAGE WITH THE BOUNDING BOXES AND THE SCORES ON IT.
    '''

    img2 = im2.copy()
    for i in range(len(New_Coordinates["X1"])):
    	#Reading coordinates
        x1pp =  New_Coordinates["X1"][i] 
        x2pp =  New_Coordinates["X2"][i] 
        y1pp =  New_Coordinates["Y1"][i] 
        y2pp =  New_Coordinates["Y2"][i] 
        cv2.rectangle(img2, (y1pp, x1pp), (y2pp, x2pp),
        (255, 0, 0) , 10)


        #Putting the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (y1pp,x1pp)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2

        cv2.putText(img2,str(scoreC[i]), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
    return img2