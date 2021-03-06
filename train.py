#!/usr/bin/env python
# coding: utf-8

# In[1]:


from osgeo import gdal

import numpy as np

import h5py
import random
from scipy import ndarray
from skimage import exposure

from random import random as rd
import cv2 as cv


import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from keras.losses import binary_crossentropy as bce
from keras.losses import categorical_crossentropy as cce
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping #,TensorBoard#, ReduceLROnPlateau, LearningRateScheduler
from keras.utils.np_utils import to_categorical  
import utils as utils


# In[2]:


def image_generator(image_path,label_path,image_id,tile_id,y_coord,x_coord, batch,isAugment=False):
    
    def raster2arr(raster, bnd,x,y,wndo):
        rr = np.zeros((wndo, wndo, bnd))
        newValue = 0
        for b in range(bnd):
            band = raster.GetRasterBand(b+1)
            img = band.ReadAsArray(x, y, wndo, wndo)
            noDataValue = band.GetNoDataValue()
            img[img == noDataValue] = newValue
            rr[:,:,b] = img
        return rr.astype(np.uint8)
    
    def com_gen(image,label,isAugment):
        K.set_image_data_format('channels_last')
        def combine_generator(gen1, gen2):
            while True:
                yield(gen1.next(), gen2.next())
        
        if isAugment is True:
            data_gen_args = dict(horizontal_flip=True,
                             vertical_flip=True,
                             rotation_range=45,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
#                              brightness_range=[0.8,1.2],
                             zoom_range=0.2,
                             fill_mode='nearest',
                             data_format = "channels_last",
                             rescale = 1./255)
        else:
            data_gen_args = dict(data_format = "channels_last",
                                 rescale = 1./255)
        
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
#         image_datagen.fit(image, augment=isAugment, rounds=1, seed=42)
#         mask_datagen.fit(label, augment=isAugment, rounds=1, seed=42)
        image_generator = image_datagen.flow(image, batch_size=1, seed=42)
        mask_generator = mask_datagen.flow(label, batch_size=1, seed=42)
        
        return combine_generator(image_generator,mask_generator)

    
    total_images = len(y_coord)
    all_images_id =list(range(total_images))
    random.seed(948)
    random.shuffle(all_images_id)
    
    current_image_id = 0
    while (current_image_id<total_images):
        
        batch_input = np.zeros((batch,size,size,3))
        batch_output = np.zeros((batch,size,size,1))
        
       
        for batch_index in range(batch):
            index = all_images_id[current_image_id]
            x = x_coord[index]
            y = y_coord[index]
            img_id = image_id[index]
            
        
            raster = gdal.Open(image_path[img_id])
            batch_input[batch_index,:,:,:] = raster2arr(raster,3,x,y,size)
            raster = None
            
            raster = gdal.Open(label_path[img_id])
            batch_output[batch_index,:,:,:]= raster2arr(raster,1,x,y,size)*255
            raster = None  
            
            current_image_id += 1
        
        return (com_gen(batch_input, batch_output,isAugment))

def dev_test_generator(image_path,label_path,image_id,tile_id,y_coord,x_coord, batch):    
    
    def raster2arr(raster, bnd,x,y,wndo):
        rr = np.zeros((wndo, wndo, bnd))
        newValue = 0
        for b in range(bnd):
            band = raster.GetRasterBand(b+1)
            img = band.ReadAsArray(x, y, wndo, wndo)
            noDataValue = band.GetNoDataValue()
            img[img == noDataValue] = newValue
            rr[:,:,b] = img
        return rr.astype(np.uint8)

    def com_gen(image2,label2):
        K.set_image_data_format('channels_last')
        def combine_generator(gen1, gen2):
            while True:
                yield(gen1.next(), gen2.next())                                   
  
        data_gen_args2 = dict(data_format = "channels_last",
                             rescale = 1./255)
        image_datagen2 = ImageDataGenerator(**data_gen_args2)
        mask_datagen2 = ImageDataGenerator(**data_gen_args2)
        #image_datagen.fit(image, augment=True, rounds=5, seed=42)
        #mask_datagen.fit(label, augment=True, rounds=5, seed=42)
        image_generator2 = image_datagen2.flow(image2, batch_size=1, seed=56)
        mask_generator2 = mask_datagen2.flow(label2, batch_size=1, seed=56)
        val_generator = combine_generator(image_generator2,mask_generator2)
        return val_generator
    
    total_images = len(y_coord)
    all_images_id =list(range(total_images))
    random.seed(948)
    random.shuffle(all_images_id)
    
    current_image_id = 0
    while (current_image_id<total_images):
        
        batch_input = np.zeros((batch,size,size,3))
        batch_output = np.zeros((batch,size,size,1))
        
       
        for batch_index in range(batch):
            index = all_images_id[current_image_id]
            x = x_coord[index]
            y = y_coord[index]
            img_id = image_id[index]
            
        
            raster = gdal.Open(image_path[img_id])
            batch_input[batch_index,:,:,:] = raster2arr(raster,3,x,y,size)
            raster = None
            
            raster = gdal.Open(label_path[img_id])
            batch_output[batch_index,:,:,:]= raster2arr(raster,1,x,y,size)*255
            raster = None  
            
            current_image_id += 1
       
        return (com_gen(batch_input, batch_output))


# In[3]:


size = 256
def unet(size, lri, input_height = size, input_width = size, nClasses = 1):

    K.set_image_data_format('channels_last')
    
    input_size = (input_width, input_height, 3)
    input1 = Input(shape = input_size)
    n = 64
    drate1 = 0.5
    drate2 = 0.5
    drate3 = 0.5
    drate4 = 0.5
    drate5 = 0.5
    lmbd = 0.0001
    #bn01 = ( BatchNormalization())(input1) 
    
    #drop0 = Dropout(0)(input1)
    conv1 = Conv2D(n, (3,3), kernel_regularizer=l2(lmbd), activation = 'relu', padding = 'same', data_format= "channels_last", kernel_initializer = 'he_normal')(input1)
    drop1 = Dropout(drate1)(conv1)
    bn11 = ( BatchNormalization())(drop1) 

    conv1 = Conv2D(n, (3,3), kernel_regularizer=l2(lmbd), activation = 'relu', padding = 'same', data_format= "channels_last", kernel_initializer = 'he_normal')(bn11)
    drop12 = Dropout(drate2)(conv1)
    bn12 = ( BatchNormalization())(drop12) 

    pool1 = MaxPooling2D(pool_size=(2, 2))(bn12)
    
    conv2 = Conv2D(n*2, (3,3), kernel_regularizer=l2(lmbd), activation = 'relu', padding = 'same',  data_format= "channels_last", kernel_initializer = 'he_normal')(pool1)
    drop2 = Dropout(drate1)(conv2) 
    bn21 = ( BatchNormalization())(drop2)

    conv2 = Conv2D(n*2, (3,3), kernel_regularizer=l2(lmbd), activation = 'relu', padding = 'same',  data_format= "channels_last", kernel_initializer = 'he_normal')(bn21)
    drop22 = Dropout(drate2)(conv2)  
    bn22 = ( BatchNormalization())(drop22)   
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn22)
    
    conv3 = Conv2D(n*4, (3,3), kernel_regularizer=l2(lmbd), activation = 'relu', padding = 'same', data_format= "channels_last", kernel_initializer = 'he_normal')(pool2)
    drop3 = Dropout(drate1)(conv3)
    bn31 = ( BatchNormalization())(drop3)

    conv3 = Conv2D(n*4, (3,3), kernel_regularizer=l2(lmbd), activation = 'relu', padding = 'same', data_format= "channels_last", kernel_initializer ='he_normal')(bn31)
    drop32 = Dropout(drate2)(conv3)
    bn32 = ( BatchNormalization())(drop32) 
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn32)
    
    conv4 = Conv2D(n*8, (3,3), kernel_regularizer=l2(lmbd), activation = 'relu', padding = 'same', data_format= "channels_last", kernel_initializer ='he_normal')(pool3)
    drop4 = Dropout(drate1)(conv4)
    bn41 = ( BatchNormalization())(drop4) 

    conv4 = Conv2D(n*8, (3,3), kernel_regularizer=l2(lmbd), activation = 'relu', padding = 'same', data_format= "channels_last", kernel_initializer ='he_normal')(bn41)
    drop42 = Dropout(drate5)(conv4)
    bn42 = ( BatchNormalization())(drop42) 
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn42)
   
    conv5 = Conv2D(n*16, (3,3), kernel_regularizer=l2(lmbd), activation = 'relu', padding = 'same', data_format= "channels_last", kernel_initializer ='he_normal')(pool4)
    drop5 = Dropout(drate4)(conv5)
    bn51 = ( BatchNormalization())(drop5) 
    conv5 = Conv2D(n*16, (3,3), kernel_regularizer=l2(lmbd), activation = 'relu', padding = 'same', data_format= "channels_last", kernel_initializer ='he_normal')(bn51)
    drop52 = Dropout(drate5)(conv5)
    bn52 = ( BatchNormalization())(drop52) 
    
    up6 = Conv2D(n*8, (2,2), kernel_regularizer=l2(lmbd), activation = 'relu', padding = 'same',data_format= "channels_last",  kernel_initializer ='he_normal')(UpSampling2D(size = (2,2))(bn52))
    merge6 = concatenate([bn42,up6], axis = 3)
    #merge6 = concatenate([conv4,up6], axis = 3)
    
    conv6 = Conv2D(n*8, (3,3), kernel_regularizer=l2(lmbd), activation = 'relu', padding = 'same', data_format= "channels_last", kernel_initializer ='he_normal')(merge6)
    drop6 = Dropout(drate3)(conv6)
    bn61 = ( BatchNormalization())(drop6) 

    conv6 = Conv2D(n*8, (3,3), kernel_regularizer=l2(lmbd), activation = 'relu', padding = 'same', data_format= "channels_last", kernel_initializer ='he_normal')(bn61)
    drop62 = Dropout(drate4)(conv6)
    bn62 = ( BatchNormalization())(drop62) 
    
    up7 = Conv2D(n*4, (2,2), kernel_regularizer=l2(lmbd), activation = 'relu', padding = 'same', data_format= "channels_last", kernel_initializer ='he_normal')(UpSampling2D(size = (2,2))(bn62))
    
    merge7 = concatenate([bn32,up7], axis = 3)
    
    
    conv7 = Conv2D(n*4, (3,3), kernel_regularizer=l2(lmbd), activation = 'relu', padding = 'same', data_format= "channels_last", kernel_initializer ='he_normal')(merge7)
    drop7 = Dropout(drate3)(conv7)
    bn71 = ( BatchNormalization())(drop7) 

    conv7 = Conv2D(n*4, (3,3), kernel_regularizer=l2(lmbd), activation = 'relu', padding = 'same', data_format= "channels_last", kernel_initializer ='he_normal')(bn71)
    drop72 = Dropout(drate4)(conv7)
    bn72 = ( BatchNormalization())(drop72) 
    

    up8 = Conv2D(n*2, (2,2), kernel_regularizer=l2(lmbd), activation = 'relu', padding = 'same', data_format= "channels_last", kernel_initializer ='he_normal')(UpSampling2D(size = (2,2))(bn72))
    
    
    merge8 = concatenate([bn22,up8], axis = 3)
    
    conv8 = Conv2D(n*2, (3,3), kernel_regularizer=l2(lmbd), activation = 'relu', padding = 'same', data_format= "channels_last", kernel_initializer ='he_normal')(merge8)
    drop8 = Dropout(drate3)(conv8)
    bn81 = ( BatchNormalization())(drop8) 

    conv8 = Conv2D(n*2, (3,3), kernel_regularizer=l2(lmbd), activation = 'relu', padding = 'same', data_format= "channels_last", kernel_initializer ='he_normal')(bn81)
    drop82 = Dropout(drate4)(conv8)
    bn82 = ( BatchNormalization())(drop82) 
    
    up9 = Conv2D(n, (2,2), kernel_regularizer=l2(lmbd), activation = 'relu', padding = 'same', data_format= "channels_last", kernel_initializer ='he_normal')(UpSampling2D(size = (2,2))(bn82))
    merge9 = concatenate([bn12,up9], axis = 3)
    
    conv9 = Conv2D(n, (3,3), kernel_regularizer=l2(lmbd), activation = 'relu', padding = 'same', data_format= "channels_last", kernel_initializer ='he_normal')(merge9)
    drop9 = Dropout(drate3)(conv9)
    bn91 = ( BatchNormalization())(drop9) 

    conv9 = Conv2D(n, (3,3), kernel_regularizer=l2(lmbd), activation = 'relu', padding = 'same', data_format= "channels_last", kernel_initializer ='he_normal')(bn91)
    drop92 = Dropout(drate4)(conv9)
    bn92 = ( BatchNormalization())(drop92) 
    
    conv10 = Conv2D(nClasses, (1,1), kernel_regularizer=l2(lmbd), activation = 'sigmoid', padding = 'same', data_format= "channels_last", kernel_initializer ='he_normal')(bn92)
    model = Model(inputs = input1, outputs = conv10)
    
    model.compile(optimizer = Adam(lr  = lri), loss= utils.dice_loss, metrics = [utils.iou, utils.dice])
   
    return model


# In[4]:


def fetch_tiles_info(ipath):
    img_id  = []
    tile_id = []
    y_coord = []
    x_coord = []


    for i in range(len(ipath)):
        img_id,tile_id,y_coord,x_coord,num_tiles = utils.apply_fetch_all_tiles(ipath[i],img_id,tile_id,y_coord,x_coord,1,i)
        
        img_id,tile_id,y_coord,x_coord = utils.apply_fetch_tiles_at_random(ipath[i],img_id,tile_id,y_coord,x_coord,num_tiles,num_tiles,i)
    
    return img_id,tile_id,y_coord,x_coord


# In[5]:


def training(model,train_generator,val_generator,steps_train,steps_val,lri = 1e-3,epoch =50,size=256,seed = 23):
             
    K.set_image_data_format('channels_last')
    
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    
    checkpoint = ModelCheckpoint("./model_1.h5", monitor='val_dice', verbose=1, save_best_only=True, save_weights_only=False, mode='max', peroid=1)
    
    early = EarlyStopping(monitor='val_dice', min_delta=0, patience=50, verbose=1, mode='max')
    history = model.fit(train_generator, steps_per_epoch = steps_train, epochs = epoch ,verbose = 2, callbacks = [checkpoint, early], validation_data= val_generator, validation_steps = steps_val)    
    


# In[ ]:




