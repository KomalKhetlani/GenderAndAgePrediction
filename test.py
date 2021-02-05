import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from model import build_model_gender
from model import build_model_age
from utils import display
import tensorflow as tf
import streamlit as st
import sys
from PIL import Image, ImageOps

st.markdown('<style>body{background-color: #f0e6e6;}</style>',unsafe_allow_html=True)

st.write("""
         # Gender and Age Group Detection
         """
         )
st.write("This is a image classification web application to predict the gender and age group of a person.")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

def import_and_predict(image_data):
    image_data.save('./TestData/uploadedImage.jpg')
    img=[]
    modelGender=tf.keras.models.load_model('./SavedGenderModel/')
    modelAge=tf.keras.models.load_model('./SavedAgeModel/')
    image = cv2.imread("./TestData/uploadedImage.jpg",0)
    image = cv2.resize(image,dsize = (64,64))
    image = image.reshape((image.shape[0],image.shape[1],1))
    img.append(image)
    features = np.array(img)
    features = features/255
    prediction = modelGender.predict(features)
    prediction1 = modelAge.predict(features)

    return prediction, prediction1

if file is None:
    st.text("")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction,prediction1 = import_and_predict(image)
    
    if prediction<=0.5 and prediction1<=0.4:
        st.write("""According to our algorithm, the uploaded picture is of an **Underage Male**""")
    elif prediction<=0.5 and prediction1>0.4 and prediction1<=1.5:
        st.write("""According to our algorithm, the uploaded picture is of an **Adult Male**""")
    elif prediction<=0.5 and prediction1>1.5:
        st.write("""According to our algorithm, the uploaded picture is of a **Male Senior Citizen**""")
    
    elif prediction>0.5 and prediction<=1 and prediction1<=0.4:
        st.write("""According to our algorithm, the uploaded picture is of an **Underage Female**""")
    elif prediction>0.5 and prediction1>0.4 and prediction1<=1.5:
        st.write("""According to our algorithm, the uploaded picture is of an **Adult Female**""")
    elif prediction>0.5 and prediction1>1.5:
        st.write("""According to our algorithm, the uploaded picture is of a **Female Senior Citizen**""")
    

    
 

