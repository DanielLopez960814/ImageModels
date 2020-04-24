import os
from tkinter import filedialog
from tkinter import Tk
import sys
from getpass import getuser


root = Tk()
root.withdraw()
dirname = filedialog.askdirectory()
print(dirname)
username = getuser()
file1 = open("C:/Users/" + username +"/AppData/Local/DatosPsophiaTool/Ruta.txt","w")#write mode 
formatos = ["JPG", "jpg", "JPEG", "jpeg", "PNG", "png"]
for filename in os.listdir(dirname):
	# do your stuff
	comprobacion = filename.split('.')[-1]
	if(comprobacion in formatos):
    		file1.write(str(dirname) + '/' + str(filename)+"\n")
	else:
			continue
    		

	

file1.close() 
root.destroy()
print("Done")
sys.stdout.flush()

	

#print('That\'\s ok, the path was already load')
#sys.stdout.flush()
#Load()