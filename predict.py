#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from osgeo import gdal
import os

import cv2 as cv
import h5py
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import utils as utils

def predict_on_image(model,batch,image,label,heights,widths,h,w,d,size=256):

    m=1
    for k in heights[0:-1]:
        p = 0
        arr = np.zeros(((len(widths)-1),size,size,d))
        img = np.zeros((size,size,d))
        for n in widths[0:-1]:
            img = image[k:(k+size), n:(n+size),:]
            arr[p,:,:,:d] = img
            p+=1
        arr = arr/255.0    
        arr = model.predict(arr, batch_size = batch, verbose = 0)

        arr = np.around(np.squeeze(arr))

        print("\r", m, '/',(len(heights)-1), end=" ")

        p = 0
        for n in widths[0:-1]:
            label[k:(k+size), n:(n+size)] = arr[p,:,:]
            p+=1
        m+=1

def adjust_height(model,batch,image,label,heights,h,w,d,size=256):
    p = 0
    arr = np.zeros(((i-1),size,size,d))
    for k in heights[0:-1]:
        img = np.zeros((size,size,d))
       
        img = image[k:(k+size), w-size:w,:]
        arr[p,:,:,:d] = img
        arr[p,:,:,:d] = img
        
        p+=1
    arr = arr/255.0    
    arr = model.predict(arr, batch_size = batch, verbose = 0)
    
    arr = np.around(np.squeeze(arr))
    print('adjusted height')
    
    p = 0
    for k in heights[0:-1]:
        label[k:(k+size), w-size:w] = arr[p,:,:]
        p+=1
    
def adjust_width(model,batch,image,label,widths,h,w,d,size=256):
    p = 0
    arr = np.zeros(((j-1),size,size,d))   
    for n in widths[0:-1]:
        img = np.zeros((size,size,d))
        img = image[h-size:h, n:(n+size),:]
        arr[p,:,:,:d] = img
        
        p+=1
    arr = arr/255.0    
    arr = model.predict(arr, batch_size = batch, verbose = 0)
    
    arr = np.around(np.squeeze(arr))
    print('adjusted width')
    p = 0
    for n in widths[0:-1]:
        label[h-size:h, n:(n+size)] = arr[p,:,:]
        p+=1
    
def batch_predict(filepath,model,batch,hi=0,wi=0,size = 256):

    raster = gdal.Open(filepath)
    w,h = round(raster.RasterXSize), round(raster.RasterYSize)
    image = (utils.raster2array(raster,3,0,0,w,h)).astype(np.uint8)
    raster = None 
    
    
    h,w,d = image.shape
    
    heights = list(range(hi,h+1,size))
    widths = list(range(wi,w+1,size))

    lHeights = len(heights)
    lWidths = len(widths)

        
    label = np.zeros((h,w))
    print('predicting')
    
    if h != size and w != size:
        predict_on_image(model,batch,image,label,heights,widths,h,w,d,size)
    
        adjust_height(model,batch,image,label,heights,h,w,d,size)

        adjust_width(model,batch,image,label,widths,h,w,d,size)
        
    
    img = (image[h-size:h, w-size:w,:])  
    arr1 = np.zeros((1,size,size,d))
    arr1[0,:,:,:d] = img
    
    arr1 = arr1/255.0    

    
    arr1 = model.predict(arr1, batch_size = 1, verbose = 0)
    
    label[h-size:h, w-size:w] = np.around(np.squeeze(arr1))
    
    return label.astype(np.uint8)


def predict_all_images(flist,mpath,batch,size = 256):
  
    K.clear_session()
    model = load_model(mpath, custom_objects={'dice_loss': utils.dice_loss,'iou': utils.iou, 'dice': utils.dice})
    

    for f, fns in enumerate(flist):
        image = batch_predict(fns,model,batch)
        h,w = image.shape[:2]
        print(image.shape)

        image2 = batch_predict(fns,model,batch,size//2,0)
        image3 = batch_predict(fns,model,batch,0,size//2)
        image4 = batch_predict(fns,model,batch,size//2,size//2)

        heights = list(range(0,h+1,size))
        widths = list(range(0,w+1,size))

        print('0: {}, 1: {}'.format(np.sum((image[image==0])+1), np.sum((image[image==1]))))

        m=1
        for k in heights[0:-1]:
            image[(k-size//4):(k+size//4),:] = image2[(k-size//4):(k+size//4),:]

        for n in widths[0:-1]:
            image[:, (n-size//4):(n+size//4)] = image3[:, (n-size//4):(n+size//4)]

        for k in heights[0:-1]:
            for n in widths[0:-1]:
                image[(k-size//4):(k+size//4),(n-size//4):(n+size//4)] = image4[(k-size//4):(k+size//4),(n-size//4):(n+size//4)]


        print('0: {}, 1: {}'.format(np.sum((image[image==0])+1), np.sum((image[image==1]))))

        image *= 255

        cv.imwrite("./predictions/prediction_{}.tif".format(f+1), (image).astype(np.uint8))              
        print('image {} saved'.format(f+1))

