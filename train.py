import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from model import build_model_gender
from model import build_model_age
from utils import display

path = "./Data/"


def create_data_labels_gender(path):
    files = os.listdir(path)
    size = len(files)
    print("Number of files available for gender classification are "+ str(size))

    images = []
    genders = []

    for file in files:
        split_var = file.split('_')
        temp_age = int(split_var[0])
        temp_gender = int(split_var[1])
        temp_ethincity = int(split_var[2])

        if (temp_ethincity !=2 and temp_age >3) or (temp_ethincity == 2 and temp_age > 5): 
            image = cv2.imread(path+file,0)
            image = cv2.resize(image,dsize = (64,64))
            image = image.reshape((image.shape[0],image.shape[1],1))
            images.append(image)
            genders.append(temp_gender)

        
    size=len(images)
    return images, genders, size

def create_data_labels_age(path):
    files = os.listdir(path)
    size = len(files)
    print("Number of files available for age classification are "+ str(size))

    images = []
    age=[]

#(0 to 10): 0
#(10 to 20):1
#(20 to 30):2
#(30 to 40):3
#(40 to 50):4
#(50 to 60):5
#(60 to 70):6
#(70 to 80):7
#(80 to 90):8
#(90 to 100):9

    for file in files:
        split_var = file.split('_')
        temp_age = int(split_var[0])
        image = cv2.imread(path+file,0)
        image = cv2.resize(image,dsize = (64,64))
        image = image.reshape((image.shape[0],image.shape[1],1))
        images.append(image)
        if (temp_age<=18):
            age.append(0)
        elif (temp_age<=60):
            age.append(1)
        else:
            age.append(2)
       
        

        
    
    return images,age,size

imageGender_list, gender_list, size_gender = create_data_labels_gender(path)
featuresGender = np.array(imageGender_list)
featuresGender = featuresGender/255
gender = np.array(gender_list)

imageAge_list,age_list,size_age=create_data_labels_age(path)
featuresAge = np.array(imageAge_list)
featuresAge = featuresAge/255
age = np.array(age_list)

modelGender = build_model_gender()
modelGender.summary()
modelGender.compile(optimizer = 'adam', loss =['mse','binary_crossentropy'],metrics=['accuracy'])
modelGender.fit(featuresGender,gender, epochs = 5, batch_size=128,shuffle = True)

print('Gender model trained')
modelGender.save('./SavedGenderModel/')
print("Gender model trained and saved")

modelAge = build_model_age()
modelAge.summary()
modelAge.compile(optimizer = 'adam', loss =['mse','binary_crossentropy'],metrics=['accuracy'])
modelAge.fit(featuresAge,age, epochs = 5, batch_size=128,shuffle = True)

print('Age model trained')
modelAge.save('./SavedAgeModel/')
print('Age model tariend and saved')