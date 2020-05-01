import os
import glob
import math
import numpy as np
import imageio
#import tensorflow as tf
from keras import optimizers
from keras import applications
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Input,Lambda, MaxPooling2D, AveragePooling2D,BatchNormalization, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt 
from keras.applications.mobilenet import MobileNet
from keras import backend as k
from PIL import Image
from keras.optimizers import RMSprop,SGD,Adam
from sklearn.utils import class_weight
from sklearn import preprocessing
from keras.regularizers import l2
from keras.layers import Activation, Add


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_image(folder, mode):
    if (mode == 'train'):
        image_files = os.listdir(folder)
    elif (mode == 'test'):
        image_files = [str(x) + '.png' for x in range(10000)]
    dataset = np.ndarray(shape=(len(image_files),28,28,3), dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        image_data = imageio.imread(image_file)
        image_data = cv2.resize(image_data,(28,28)).astype(np.float32)
        image_data = image_data/255.0
        image_data = (image_data-0.1307)/0.3081
        image_data = (np.repeat(image_data[..., np.newaxis], 3, -1))
        dataset[num_images, :, :] = image_data
        num_images += 1
            
    dataset = dataset[0:num_images, :, :]

    print('Full dataset tensor:', dataset.shape)

    return dataset

input_data_dir = './hw4_train'
classes = sorted([o for o in os.listdir(input_data_dir)]) #get classes

train_path = [input_data_dir + '/' + x for x in os.listdir(input_data_dir)]
train_data = np.array([image for d in train_path for image in load_image(d,'train')])
train_label = np.array([int(d[-1]) for d in train_path for image in load_image(d,'train')])



train_x, val_x, train_y, val_y = train_test_split(train_data, train_label, test_size=0.2)#random_state=42)


nb_train_samples = len(train_x)
nb_validation_samples = len(val_x)

train_y = np_utils.to_categorical(train_y, 10)
val_y = np_utils.to_categorical(val_y, 10)



val_datagen = ImageDataGenerator(data_format='channels_last')
#val_datagen.fit(val_x)
val_data = val_datagen.flow(x=val_x, y=val_y, batch_size=128, shuffle=False)

def random_erase(input_img):
    #code of random erasing is adapted from Z. Zhong, L. Zheng, G. Kang, S. Li, and Y. Yang, "Random Erasing Data Augmentation," 
    #https://github.com/yu4u/cutout-random-erasing
    height, width, channel = input_img.shape
    prob_erase = np.random.rand()
    if prob_erase > 0.5:
        return input_img
    while True:
        para = np.random.uniform(0.3,1/0.3)
        area = np.random.uniform(0.02,0.4)*height*width
        new_height = int(np.sqrt(area*para))
        new_width = int(np.sqrt(area/para))
        left = np.random.randint(0,width)
        top = np.random.randint(0,height)
        
        if left+new_width <= width and top+new_height <= height:
            break
    eraser = np.random.uniform(0,1)
    input_img[top:top+new_height,left:left+new_width,:] = eraser
    return input_img


aug_datagen = ImageDataGenerator(preprocessing_function = random_erase, data_format='channels_last',horizontal_flip=True)
#aug_datagen.fit(train_x)
train_data = aug_datagen.flow(x=train_x, y=train_y, batch_size=128, shuffle=True)

#input_tensor = Input(shape=(96, 96, 3))
#base_model = applications.MobileNetV2(include_top=False,weights=None,input_tensor=input_tensor, input_shape=(96,96,3),pooling='max',classes=len(classes))
#for layer in base_model.layers:
#    layer.trainable = True
#model = Flatten()(base_model.output)
#model = Dense(1024, activation='relu', name='fc1')(base_model.output)
#model = Dropout(0.5)(base_model.output)
#model = Dense(128, activation='relu', name='fc2')(model)
#model = Dropout(0.5)(model)
#predict = Dense(len(classes), activation='softmax')(model)

def make_block(in_layer,out_layer,num_layer,strides,dropRate, channel_axis):
    if in_layer != out_layer:
        x = BatchNormalization(axis = channel_axis)(in_layer)
        x = Activation("relu")(x)
        x_res = x
    else:
        x_res = BatchNormalization(axis = channel_axis)(out_layer)
        x_res = Activation("relu")(x_res)
    x_res = Conv2D(out_layer, (3,3), strides=strides, padding="same",use_bias = False)(x_res)
    x_res = BatchNormalization(axis=channel_axis)(x_res)
    x_res = Activation("relu")(x_res)
    if dropRate:
        x_res = Dropout(dropRate)(x_res)
    x_res = Conv2D(out_layer, (3,3), strides=1, padding="same",use_bias = False)(x_res)
    if in_layer != out_layer:
        shortCut = Conv2D(out_layer, (1,1), strides=1, padding="valid",use_bias = False)(x_res)
        x = Add()([x_res,shortCut])
    x = Add()([x_res,x])
    
    for i in range(num_layer):
        
        x_res = BatchNormalization(axis = channel_axis)(x)
        x_res = Activation("relu")(x_res)
        x_res = Conv2D(out_layer, (3,3), strides=strides, padding="same",use_bias = False)(x_res)
        if dropRate:
            x_res = Dropout(dropRate)(x_res)
        x_res = BatchNormalization(axis = channel_axis)(x)
        x_res = Activation("relu")(x_res)
        x_res = Conv2D(out_layer, (3,3), strides=1, padding="same",use_bias = False)(x_res)
        if in_layer != out_layer:
            shortCut = Conv2D(out_layer, (1,1), strides=1, padding="valid",use_bias = False)(x_res)
            x = Add()([x_res,shortCut])
        x = Add()([x_res,x])
    
    return x


def build_model(input_dim, output_dim, n, widen_factor, dropRate, channel_axis, weight_decay):
    assert (n-4)%6 == 0
    assert widen_factor%2 == 0
    n = (n-4)//6 
    
    nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
    
    inputs = Input(shape = input_dim)
    conv1 = Conv2D(nChannels[0], (3,3), strides = 1, padding="same", use_bias = False)(inputs)
    
    w_conv2 = make_block(in_layer = conv1, out_layer=nChannels[1], num_layer=n, strides=1, dropRate=dropRate, channel_axis=channel_axis)
    w_conv3 = make_block(in_layer = w_conv2, out_layer=nChannels[2], num_layer=n, strides=2, dropRate=dropRate, channel_axis=channel_axis)
    w_conv4 = make_block(in_layer = w_conv3, out_layer=nChannels[3], num_layer=n, strides=2, dropRate=dropRate, channel_axis=channel_axis)
    
    norm5 = BatchNormalization(axis=channel_axis)(w_conv4)
    relu6 = Activation("relu")(norm5)
    
    pool7 = AveragePooling2D(pool_size=(7, 7))(relu6)
    flatten = Flatten()(pool7)
    outputs = Dense(output_dim, activation="softmax", use_bias=False, W_regularizer=l2(weight_decay))(flatten)
    
    model = Model(input=inputs, output=outputs)
    
    return model
    
    

#mobileNet_model = Model(inputs=input_tensor, outputs=predict)
#mobileNet_model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01 / 70, amsgrad=True), loss='categorical_crossentropy', metrics=['accuracy'])
#mobileNet_model.summary()

weight_decay = 0.0005
sgd = SGD(lr=0.1, momentum=0.9, decay=weight_decay)
channel_axis = -1
input_dim = (28, 28, 3)
#Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01 / 70, amsgrad=True),

mobileNet_model = build_model(input_dim=input_dim, output_dim=len(classes), n=28, widen_factor=10, dropRate=0, channel_axis=channel_axis, weight_decay=weight_decay)
mobileNet_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
mobileNet_model.summary()

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique([y.argmax() for y in train_y]),
                                                 [y.argmax() for y in train_y])

early_stopping = EarlyStopping(verbose=1, patience=30, monitor='val_loss')
model_checkpoint = ModelCheckpoint(filepath='./models/mobileNet_checkpoints.h5', verbose=1, save_best_only=True, monitor='val_acc')
callbacks = [model_checkpoint]

history_fit = mobileNet_model.fit_generator(train_data, steps_per_epoch=nb_train_samples/128.0, epochs=300,
                    validation_data=val_data, validation_steps=nb_validation_samples/128.0, callbacks=callbacks,class_weight=class_weights)


model_json=mobileNet_model.to_json()
with open("./models/mobileNet_json.json", "w") as json_file:
    json_file.write(model_json)
mobileNet_model.save_weights("./models/mobileNet_weights.h5")
mobileNet_model.save("./models/mobileNet_model.h5")
