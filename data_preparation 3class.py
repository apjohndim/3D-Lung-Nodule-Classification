import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import pylidc as pl
import os 
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import random
import cv2
from PIL import Image
patID = []

#%%
scans = pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= 5,
                                 pl.Scan.pixel_spacing <= 5) #load all scans in object scan
print(scans.count()) #print how many scans have the particular chracterisics: slice_thickness and pixel_spacing

path = 'J:\\PHD + MSC DATASETS\\Datasets\\LIDC DATABASE\\LIDC-IDRI\\'
path_folders = os.listdir(path)

path2 = 'J:\\PHD + MSC DATASETS\\Datasets\\LIDC DATABASE\\3d\\'
nodule_count = 0
my_id = 1
g = 10000
for pid in path_folders [822:]: #[1:10]:

    #pid = 'LIDC-IDRI-0078' #select a specific folder name with scans
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).all()
    patID.append(pid)
    
    print ('[INFO] FOUND %4d SCANS' % len(scans))
    for scan in scans:#scan object of this pid
       
       ann = scan.annotations
       
       try:
           vol = scan.to_volume()
       except ValueError:
           continue
       except OSError as err:
           continue
       nods = scan.cluster_annotations()
       #anns = nods[0]

       print(len(scan.annotations)) #how many annotations
       print("'[INFO] %s has %d nodules." % (scan, len(nods)))
       
       it = len(nods)
       for nodule in range(0,it):
           #my_id = my_id+1
           y = nods[nodule]
           x = y[0]#grab the first annotation
           
           ano = len(y)
           mal = 0
           for i in range(0,ano):
               o = y[i]
               a1 = o.malignancy
               mal = mal+a1
           
           MalMean = mal/ano
           print (MalMean)  
           slices = x.contour_slice_indices
           place = x.contours_matrix[0]
           xc = place[0]
           yc = place[1]
           slide = slices[int(len(slices)/2)]
           #for slide in slices:
           nodule_count = nodule_count + 1  
           
           for k in range(-5,5):
               try:
                   vol1 = vol[:,:,slide+k]
               except IndexError:
                   vol1 = vol[:,:,slide + (abs(k)-4)]
                   
               vol1 = vol1[(xc-25):(xc+25),(yc-25):(yc+25)]
               vol1 = (vol1 - np.amin(vol1))/np.amax(vol1) #bring the pixel values to 0,1
               vol1 = vol1*255                        # pixel values set to 0,255
               im = Image.fromarray(vol1)
               im = im.convert("L")                   
               if MalMean <= 2.25:
                   lab = 'benign'
                #name = path + pid + '\\' + lab + str(g) + str('Series:') + str(my_id) + '.tif'
                   
                   if not os.path.exists('C:\\Users\\User\\SPN3D_1\\LIDC3D\\benign\\' +str(g)+ '\\'):
                          os.mkdir('C:\\Users\\User\\SPN3D_1\\LIDC3D\\benign\\' +str(g)+ '\\')
                   namepath ='C:\\Users\\User\\SPN3D_1\\LIDC3D\\benign\\' +str(g)+ '\\'
                   im.save(namepath + str('Nod_') + str(nodule_count) + 'Slice_' + str(k+5) + 'Patient number_' + str(my_id) + '.tif')
               elif MalMean >= 3.5:
                   lab='malignant'
                   
                   if not os.path.exists('C:\\Users\\User\\SPN3D_1\\LIDC3D\\malignant\\' +str(g)+ '\\'):
                       os.mkdir('C:\\Users\\User\\SPN3D_1\\LIDC3D\\malignant\\' +str(g)+ '\\')
                   namepath ='C:\\Users\\User\\SPN3D_1\\LIDC3D\\malignant\\' +str(g)+ '\\'
                   im.save(namepath + str('Nod_') + str(nodule_count) + 'Slice_' + str(k+5) + 'Patient number_' + '.tif')
               else:
                   if not os.path.exists('C:\\Users\\User\\SPN3D_1\\LIDC3D\\unl\\' +str(g)+ '\\'):
                       os.mkdir('C:\\Users\\User\\SPN3D_1\\LIDC3D\\unl\\' +str(g)+ '\\')
                   namepath ='C:\\Users\\User\\SPN3D_1\\LIDC3D\\unl\\' +str(g)+ '\\'
                   
                   if MalMean<3: 
                       lab = 'SuspBen' 
                   else: lab='SuspMal'
                   
                   im.save(namepath + str('Nod_') + str(nodule_count) + 'Slice_' + str(k+5) + 'Patient number_' + str(my_id)+ str(lab)  + '.tif')
                   
           g = g + 1
           my_id = my_id + 1     

       
print ('[INFO] EXTRACTED %4d SCANS' % nodule_count)






