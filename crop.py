# Capture the mouse click events in Python and OpenCV
'''
-> draw shape on any image 
-> reset shape on selection
-> crop the selection
run the code : python capture_events.py --image image_example.jpg
'''


# import the necessary packages
import cv2
import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from getpass import getuser

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not


ref_point = [] #variable that indicates de coordinates
cropping = False
username = getuser()
def Cropping():


	
	def shape_selection(event, x, y, flags, param):
	  # grab references to the global variables
	  global ref_point, cropping

	  # if the left mouse button was clicked, record the starting
	  # (x, y) coordinates and indicate that cropping is being
	  # performed
	  if event == cv2.EVENT_LBUTTONDOWN:
		  ref_point = [(x, y)]
		  cropping = True

	  # check to see if the left mouse button was released
	  elif event == cv2.EVENT_LBUTTONUP:
		  # record the ending (x, y) coordinates and indicate that
		  # the cropping operation is finished
		  ref_point.append((x, y))
		  cropping = False

		  # draw a rectangle around the region of interest
		  cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
		  cv2.imshow("image", image)
		  #print('ref_point: ' + str(ref_point))
		
		
	# load the image, clone it, and setup the mouse callback function
	#image = cv2.imread("prueba.jpg")

	#DEFINING THE PATH
	path = 'C:/Users/' + username +'/AppData/Local/DatosPsophiaTool/Ruta.txt' 
	print(path)

	# SELECTING RANDOM PICTURE
	f = open(path, "r").readlines() 
	Pic = f[ random.randint(0, len(f)-1 ) ]
	Pic = Pic.strip('\n') # removing '\n': enter


	cap = cv2.VideoCapture(Pic)
	# Read first frame
	_, prev = cap.read() 
	cap.release()

	#LOADING THE IMAGE
	image = prev

	#SAVING THE ORIGINAL IMAGE
	Im1 = image #original
	print(Im1.shape)
	fil_O, col_O, cap_O = Im1.shape

	#RESIZING THE IMAGE
	image = cv2.resize(image, (1200, 700)) #standard
	clone = image.copy()
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", shape_selection)

	# keep looping until the 'q' key is pressed
	while True:
	  # display the image and wait for a keypress
	  cv2.imshow("image", image)
	  key = cv2.waitKey(1) & 0xFF

	  # if the 'r' key is pressed, reset the cropping region
	  if key == ord("r"):
	  	image = clone.copy()

	  # if the 'c' key is pressed, break from the loop
	  elif key == ord("c"):
	  	break

	# if there are two reference points, then crop the region of interest
	# from teh image and display it
	if len(ref_point) == 2:
		print(ref_point[0][1])
		print(ref_point[1][1])
		crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
		cv2.imshow("crop_img", crop_img)
		cv2.waitKey(0)
	
	y_i = int((ref_point[0][0]*(col_O/1200)))
	y_f = int((ref_point[1][0]*(col_O/1200)))
	x_i = int((ref_point[0][1]*(fil_O/700)))
	x_f = int((ref_point[1][1]*(fil_O/700)))

	final = [(x_i,y_i), (x_f,y_f)]
	#plt.imshow(Im1[x_i:x_f,y_i:y_f,:]), plt.show()
	#final = [(ref_point[0][0],ref_point[0][1]), (ref_point[1][0],ref_point[1][1])]
	# close all open windows
	cv2.destroyAllWindows()
	f= open("C:/Users/" + username +"/AppData/Local/DatosPsophiaTool/coordinatesCropping.txt","w+")
	f.write(str(final))
	f.close()
	print("Done")
	sys.stdout.flush()
	
Cropping()
print('The images was cropped correctly')
sys.stdout.flush()