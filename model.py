import keras 
from keras.layers import *
from keras.models import *
from keras import backend as K

def build_model_gender():
    inputs = Input(shape=(64,64,1))
    conv1 = Conv2D(32, kernel_size=(3, 3),activation='relu')(inputs)
    conv2 = Conv2D(64, kernel_size=(3, 3),activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, kernel_size=(3, 3),activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, kernel_size=(3,3), activation = 'relu')(pool2)
    conv5 = Conv2D(256, kernel_size=(3,3), activation = 'relu')(conv4)
    flat = Flatten()(conv5)
    gender_conv1 = Dense(256, activation='relu')(flat)
    gender_conv2 = Dense(128, activation='relu')(gender_conv1)
    gender_conv3 = Dense(64, activation='relu')(gender_conv2)
    gender_conv4 = Dense(32, activation='relu')(gender_conv3)
    gender_conv5 = Dense(16, activation='relu')(gender_conv4)
    gender_conv6 = Dense(8, activation='relu')(gender_conv5)
    gender_model = Dense(1, activation='sigmoid')(gender_conv6)

    return Model(inputs,gender_model)

def build_model_age():
    inputs = Input(shape=(64,64,1))
    conv1 = Conv2D(32, kernel_size=(3, 3),activation='relu')(inputs)
    conv2 = Conv2D(64, kernel_size=(3, 3),activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, kernel_size=(3, 3),activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, kernel_size=(3,3), activation = 'relu')(pool2)
    conv5 = Conv2D(256, kernel_size=(3,3), activation = 'relu')(conv4)
    flat = Flatten()(conv5)
    age_conv1 = Dense(256, activation='relu')(flat)
    age_conv2 = Dense(128, activation='relu')(age_conv1)
    age_conv3 = Dense(64, activation='relu')(age_conv2)
    age_conv4 = Dense(32, activation='relu')(age_conv3)
    age_conv5 = Dense(16, activation='relu')(age_conv4)
    age_conv6 = Dense(8, activation='relu')(age_conv5)
    age_model = Dense(1, activation='relu')(age_conv6)

    return Model(inputs,age_model)


if __name__ == "__main__":
    modelGender = build_model_gender()
    modelGender.summary()
    modelAge = build_model_age()
    modelAge.summary()