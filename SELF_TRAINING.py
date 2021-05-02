
print("[INFO] Importing Libraries")
import matplotlib as plt
plt.style.use('ggplot')
# matplotlib inline
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import time   # time1 = time.time(); print('Time taken: {:.1f} seconds'.format(time.time() - time1))
import warnings
import keras
from keras.preprocessing.image import ImageDataGenerator
warnings.filterwarnings("ignore")
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Conv3D, MaxPooling3D
from keras.utils import to_categorical
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import regularizers
from keras import optimizers
from keras.layers import LeakyReLU
from keras.layers import ELU
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D, ZeroPadding3D, GlobalAveragePooling3D, GlobalMaxPooling3D
from PIL import Image 
import numpy
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import time
from sklearn.metrics import classification_report, confusion_matrix
from keras_applications.resnet import ResNet50
from keras_applications.mobilenet import MobileNet
import tensorflow as tf
SEED = 50   # set random seed
print("[INFO] Libraries Imported")
from copy import deepcopy
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.models import load_model
import custom1

#adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
#leakyrelu = keras.layers.LeakyReLU(alpha=0.3)
#elu = keras.layers.ELU(alpha=1.0)




def conv3D (x,filters,bn,maxp=0,rdmax=0,drop=True,DepthPool=False):
    
    x  = Conv3D(filters, (3, 3,3), padding='same', activation='selu')(x)
    
    
    if maxp ==1 and DepthPool==False:
        x = MaxPooling3D((1,2, 2),padding='valid')(x)
    if  DepthPool==True:   
        x = MaxPooling3D((2,2, 2),padding='valid')(x)
        
    if drop==True:
        x = Dropout(0.5)(x)
        
    if bn==1:
        x = BatchNormalization(axis=-1)(x)
        
    if rdmax == 1:
        x = MaxPooling3D((2,2, 2),padding='valid')(x)
    return x

#%%

def make_3d():
    input_img = Input(shape=(6, 32, 32, 3)) 
    
    x = conv3D (input_img,filters=64,bn = 1, maxp=0,rdmax=0,drop=False, DepthPool=True)
    x = conv3D (x,64,bn = 1, maxp=0,rdmax=0,drop=False, DepthPool=False)
    
    y = conv3D (x,128,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False)
    y = conv3D (y,128,bn = 1, maxp=0,rdmax=0,drop=False, DepthPool=True)
    #y = conv3D (y,64,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False)
    
    z = conv3D (y,256,bn = 1, maxp=0,rdmax=0,drop=False, DepthPool=False)
    z = conv3D (z,256,bn = 1, maxp=0,rdmax=0,drop=False, DepthPool=False)
    z = conv3D (z,256,bn = 1, maxp=0,rdmax=0,drop=False, DepthPool=False)
    z = conv3D (z,256,bn = 1, maxp=0,rdmax=0,drop=False, DepthPool=False)
    
    
    # w = conv3D (z,512,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False)
    # w = conv3D (w,512,bn = 1, maxp=0,rdmax=0,drop=False, DepthPool=False)
    # w = conv3D (w,512,bn = 1, maxp=0,rdmax=0,drop=False, DepthPool=False)
    # w = conv3D (w,512,bn = 1, maxp=0,rdmax=0,drop=False, DepthPool=False)
    #k = conv3D (w,128,0,0)
    #m = conv3D (l,1024,bn = 1, maxp=0,rdmax=0,drop=False, DepthPool=False)
    
    #o = conv3D (m,512,1,1)
    #i = conv3D (o,512,1,0)
    
    #a = GlobalAveragePooling3D()(y)
    #b = GlobalAveragePooling3D()(z)
    c = GlobalAveragePooling3D()(z)
    #d = GlobalAveragePooling3D()(l)
    #e = GlobalAveragePooling3D()(i)
    #f = GlobalAveragePooling3D()(m)
    
    #n = keras.layers.concatenate([c,b,d], axis=-1)
    
    
    
    n = Dense(4096, activation='selu')(c)
    n = Dropout(0.5)(n)
    # n = Dense(4096, activation='selu')(c)
    # n = Dropout(0.5)(n)
    #n = Dense(750, activation='elu')(n)
    #n = Dropout(0.5)(n)
    n = Dense(2, activation='softmax')(n)
    model = Model(inputs=input_img, outputs=n)
    
    #opt = SGD(lr=0.01)
    
    opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    
#import pydot

    return model

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]

def load3d (data_path):

    k = 0
    classes = os.listdir(data_path) #list the subfolders of the main path (subfolders must be the classes)
    data = []
    #data3 = np.array([])
    labels = []
    for classe in classes: #for each class
    
        nodules = os.listdir(data_path+classe+'\\')
        
        for nodule in nodules:
            
            imagePaths = sorted(list(paths.list_images(data_path + str('\\') + str(classe)+ '\\' + nodule))) #list the folders inside the class. each folder contains 10 slices
    
            for imagePath in imagePaths: #for each slices_folder
                
                image = cv2.imread(imagePath)
                image = cv2.resize(image, (32, 32))/255
                data.append(image) #create a 3d array with the images  
                label = imagePath.split(os.path.sep)[-3] #take the label from the initial folder
            k = k + 1
            data2 = np.array(data, dtype="float")   
            data2 = data2.reshape(data2.shape[0], 32, 32, 3) 
            data2 = data2[None] #add a 5th dimension
    
            if k ==1: 
                data3 = np.copy(data2)
                data2 = []
                data = []
                labels.append(label)
                continue
            labels.append(label)
            data3 = np.append(data3, data2, axis=0)
            data2 = []
            data = []
            #print ('3D Image obtained')
            #data3 = np.concatenate([data3,data2], axis=0) #append based on the 5th dimension, which is the batch size
        
    labels = np.array(labels) 
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')  
    print ("[INFO] OBTAINED 2D IMAGES AND BATCHED TO 3D IMAGES ARRAY OF 5 DIMENSIONS.")
    
    data3, labels = unison_shuffled_copies(data3, labels)
    
    
    return data3, labels
    



def pick_reliable(prediction_num, fake_data, threshold,numben,nummal):#pick the most reliable predictions and return them with the images. returns new dataU
    data=[]
    labels=[]
    dataU = []
    new1 = 0
    new2 = 0
    diafora = numben-nummal
    if diafora <=0:
        print('[INFO] Pick_reliable_Feedback: The Malignant nodules are %4d more than the benign' %abs(diafora))
        issue = 'mal more'
    else: 
        print('[INFO] Pick_reliable_Feedback: The Benign nodules are %4d more than the benign' %abs(diafora))
        issue = 'ben more'
    
    diafora = abs(diafora)
    
    for i in range(len(fake_data)):
        if prediction_num[i,0]>threshold and numben+new1<nummal+new2+50 :

                    f = fake_data[i,:,:,:]
                    data.append(f)
                    label = 'benign'
                    new1 = new1+1
                    labels.append(label)
                
        elif prediction_num[i,1]>threshold:
                f = fake_data[i,:,:,:]
                data.append(f)
                label = 'malignant'
                labels.append(label)
                new2 = new2+1
                
        else: 
            f = fake_data[i,:,:,:]
            dataU.append(f)
    
            
    print ('[INFO] Pick_reliable_Feedback: Added %4d benign nodules' %(new1) )
    print ('[INFO] Pick_reliable_Feedback: Added%4d malignant nodules' %(new2) )
    print ('[INFO] Pick_reliable_Feedback: Picked a total of {} reliable predictions with threshold set to {}'.format(len(data), threshold))
    data = np.array(data, dtype="float")
    dataU = np.array(dataU, dtype="float") 
    labels = np.array(labels)
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    #labels = np.argmax(labels, axis=-1)
    return data, labels, dataU
            
       



         
def merge_datas(data, labels, datanext, labelsnext):#merge the previous training data and labels with the new
    
    mergeX = np.concatenate([data, datanext])
    mergeY = np.concatenate([labels, labelsnext])
    print ('[INFO] Merger: The datas were merged. The new size of the data I return is: {}'.format(len(mergeY)))
    return mergeX, mergeY
  



  
es = EarlyStopping(monitor='val_loss', mode='auto',patience=5, verbose=1)
    
def train_self (dataL, dataU, labelsL, epochs):#train the model on any labelled data. returns the accuracy and the predicitions on unlabelled data
    
    
    model3 = load_model('C:\\Users\\User\\SPN3D_1\\models\\3dvggd12v45.h5')


    final_score = 0.85
    predict_fake = model3.predict(dataU)

    return final_score, predict_fake   


    
def train_self2 (dataL1, dataU, labelsL1, dataL, labelsL, best_pred, pred_labels, epochs,i):#train the model on any labelled data. returns the accuracy and the predicitions on unlabelled data
    
    data = dataL1
    labels = labelsL1
    
    
    #data_ext,labels_ext = load_images('')
    

    
    n_split=10 #10fold cross validation
    scores = [] #here every fold accuracy will be kept
    predictions_all = np.empty(0) # here, every fold predictions will be kept
    predictions_all_num = np.empty([0,2]) # here, every fold predictions will be kept
    test_labels = np.empty(0) #here, every fold labels are kept
    name2 = 5000 #name initiator for the incorrectly classified insatnces
    conf_final = np.array([[0,0],[0,0]]) #initialization of the overall confusion matrix
    for train_index,test_index in KFold(n_split).split(data):
        trainX,testX=data[train_index],data[test_index]
        trainY,testY=labels[train_index],labels[test_index]
        
        

        
        if i==0:
                  trainX = np.concatenate([best_pred, trainX])
                  trainY = np.concatenate([pred_labels, trainY])  
        else:    
            
            trainX = np.concatenate([trainX, dataL, best_pred])
            trainY = np.concatenate([trainY, labelsL, pred_labels])
        
        
        

        model3 = make_3d()
    
        aug = custom1.customImageDataGenerator(rotation_range=45,
                 width_shift_range=0.01,
                 height_shift_range=0.01,
                 horizontal_flip=True, vertical_flip=True)
    
        model3.fit_generator(aug.flow(trainX, trainY,batch_size=64), epochs=10, steps_per_epoch=len(trainX)//64)
    

        #model3.fit(trainX, trainY ,epochs=epochs, batch_size=64)
        time.sleep(1) 
        score = model3.evaluate(testX,testY)
        score = score[1] #keep the accuracy score, not the loss
        scores.append(score) #put the fold score to list
        testY2 = np.argmax(testY, axis=-1) #make the labels 1column array
        print('Model evaluation ',score)
        predict = model3.predict(testX) #for def models functional api
        predict_num = predict
        predict = predict.argmax(axis=-1) #for def models functional api
        conf = confusion_matrix(testY2, predict) #get the fold conf matrix
        conf_final = conf + conf_final #sum it with the previous conf matrix
        name2 = name2 + 1
        predictions_all = np.concatenate([predictions_all, predict]) #merge the two np arrays of predicitons
        predictions_all_num = np.concatenate([predictions_all_num, predict_num])
        test_labels = np.concatenate ([test_labels, testY2]) #merge the two np arrays of labels
    scores = np.asarray(scores)
    final_score = np.mean(scores)  
    #model3.save('self_vgg19fe.h5')
    
    time.sleep(2) 
    predict_fake = model3.predict(dataU)
    time.sleep(5) 
    model3.save('self_trained3D.h5')
    return final_score, predict_fake, conf_final, model3



#%% train self

#load labelled data

path = 'C:\\Users\\User\\SPN3D_1\\LIDC3D\\'
dataL1,labelsL1 = load3d(path)

#load unlabelled data initial
path2 = 'C:\\Users\\User\\SPN3D_1\\datau\\'
dataU1,labelsU1 = load3d(path2)

#path3 = 'C:\\Users\\User\\gan_classification_many datasets\\PET'
#datahidden,labelshidden = load_images(path3)
# how many attempts to concatenate new instances
iterations = 10
#%%
#self training
epochs = 6
threshold = 0.97
dataL = deepcopy(dataL1[:2])
labelsL = deepcopy(labelsL1[:2])
dataU = dataU1




#%%

for i in range (iterations):
    
    print('[INFO] Main Oberver: Iteration Number {}'.format(i+1))
    if i==0:
        final_score, predict_num = train_self (dataL1, dataU, labelsL1, epochs)
        print('[INFO] Main Oberver: Accuracy on Labelled set: {}'.format(final_score)) 
        old_score = 0.78
        how_many = [sum(x) for x in zip(*labelsL1)]
        how_benign = deepcopy(np.array(int((how_many[0]))))
        how_malignant = deepcopy(np.array(int((how_many[1]))))
    else:
        how_many = [sum(x) for x in zip(*labelsL)]
        how_benign = deepcopy(np.array(int((how_many[0]))))
        how_malignant = deepcopy(np.array(int((how_many[1]))))
    
       
    best_pred, pred_labels, dataU = pick_reliable(predict_num,dataU,threshold, how_benign, how_malignant)
    best_pred, pred_labels = unison_shuffled_copies(best_pred, pred_labels)


    
    print('[INFO] Main Observer: The picker function returned {} nodules'.format(len(pred_labels)))
    
    if i==0:
        dataL = deepcopy(best_pred)
        labelsL = deepcopy(pred_labels)

    #dataL, labelsL = merge_datas(dataL,labelsL, best_pred, pred_labels)
    op = len(dataL1)+len(best_pred)
    print('[INFO] Main Observer: Labelled Training Data increased to {}'.format(op))
    print('[INFO] Main Observer: Training with the expanded data...')

    #with tf.device('/cpu:0'):        
    final_score, predict_num, conf_final,model3 = train_self2 (dataL1, dataU, labelsL1, dataL, labelsL, best_pred, pred_labels, epochs,i)
   
    print('[INFO] New Accuracy: {}'.format(final_score))
    
    if old_score < final_score:
        dataL, labelsL = merge_datas(dataL,labelsL, best_pred, pred_labels)
        dataL, labelsL = unison_shuffled_copies(dataL, labelsL)
        dataL1, labelsL1 = unison_shuffled_copies(dataL1, labelsL1)
        old_score = final_score
        model3.save('3Dlast.h5')
        print('[INFO] Merging the newly labelled data to a different labelled set...')
        print('[INFO] The initial datasize is: %4d, the new labelled data size is: %5d , remaining unlabelled instances: %5d' %(len(dataL1), len(dataL)+len(dataL1), len(dataU)))
    else: print ('[INFO] Accuracy dropped. The new picks are removed. The new labelled data size is: ' + str (len(dataL)) + '. The old labelled data size is: ' + str(len(dataL1)))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
    
    