#!/usr/bin/env python
# coding: utf-8

# In[1]:


from osgeo import gdal
import numpy as np
import cv2 as cv
import random
import os
import re
import tensorflow as tf
from tensorflow.keras import backend as K


def shuffle_list(*ls, seeds =78):
  random.seed(seeds)
  l =list(zip(*ls))

  random.shuffle(l)
  return zip(*l)

def raster2array(raster, bnd,x,y,wndo):
    rr = np.zeros((wndo, wndo, bnd))
    newValue = 0
    for b in range(bnd):
        band = raster.GetRasterBand(b+1)
        rr[:,:,b] = band.ReadAsArray(x, y, wndo, wndo)
        noDataValue = band.GetNoDataValue()
    rr[rr == noDataValue] = newValue
    return rr.astype(np.uint8)

def raster2array(raster, bnd,x,y,wndox,wndoy):
    rr = np.zeros((wndoy, wndox, bnd))
    newValue = 0
    for b in range(bnd):
        band = raster.GetRasterBand(b+1)
        rr[:,:,b] = band.ReadAsArray(x, y, wndox, wndoy)
        noDataValue = band.GetNoDataValue()
    rr[rr == noDataValue] = newValue
    return rr

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
  
  
def get_all_images(folder, ext):

    all_files = []
    #Iterate through all files in folder
    for file in sorted_aphanumeric(os.listdir(folder)):
        #Get the file extension
        _,  file_ext = os.path.splitext(file)

        #If file is of given extension, get it's full path and append to list
        if ext in file_ext:
            full_file_path = os.path.join(folder, file)
            all_files.append(full_file_path)

    #Get list of all files
    return all_files

def fetch_all_tiles(ipath, image_id, hmargin, wmargin, steps,size = 256):
    
    
    raster = gdal.Open(ipath)
    w,h = round(raster.RasterXSize), round(raster.RasterYSize)
    raster = None
    
    
    y_coord= (np.arange(hmargin,h+1,size))
    y_coord = y_coord[:-1]
    
    x_coord= np.arange(wmargin,w+1,size)
    x_coord = x_coord[:-1]
    

    i= (y_coord.shape[0]) 
    j = (x_coord.shape[0])
    num_tiles = i*j
    
    print("image: {}, number of tiles: {}".format(image_id,num_tiles))
    
    x_coord = x_coord*np.ones((i,j))
    y_coord = y_coord.reshape(i,1)*(np.ones((i,j)))
    
    tile_id = np.arange(0,num_tiles,1).reshape(i,j)
    img_id = np.ones_like(tile_id) * image_id 
    y_coord = np.around(y_coord)
    x_coord = np.around(x_coord)


    step = steps
    tile_id = list(tile_id.ravel()[::step])
    y_coord = list(y_coord.ravel()[::step])
    x_coord = list(x_coord.ravel()[::step])
    img_id = list(img_id.ravel()[::step])

    return img_id,tile_id,y_coord,x_coord

def apply_fetch_all_tiles(filepath,img_id,tile_id,y_coord,x_coord,steps, unique_id):
    
    raster = gdal.Open(filepath)
    w,h = round(raster.RasterXSize), round(raster.RasterYSize)
    
    raster = None

    img_id1,tile_id1,y_coord1,x_coord1 = fetch_all_tiles(filepath,unique_id,0,0,steps)     
    
    img_id.extend(img_id1)
    tile_id.extend(tile_id1)
    y_coord.extend(y_coord1)
    x_coord.extend(x_coord1)
    
    return img_id,tile_id,y_coord,x_coord,len(img_id1)



def fetch_tiles_at_random(start_w, start_h, w,h, wmargin, hmargin, starting_tile_id ,number_of_tiles, image_id):
    
    np.random.seed(440+image_id)
    y_coord = list(np.squeeze(np.random.randint(start_h,(h-hmargin),size=(number_of_tiles,1)).astype(np.float64)))
    np.random.seed(494+image_id)
    x_coord = list(np.squeeze(np.random.randint(start_w,(w-wmargin),size=(number_of_tiles,1)).astype(np.float64)))
    tile_id = list(np.squeeze(np.arange(starting_tile_id,starting_tile_id+number_of_tiles,1).reshape(number_of_tiles,1)))
    img_id = list(np.squeeze(np.zeros((number_of_tiles,1),dtype=np.uint8)+image_id))
    print("image: {}, number of random crops: {}".format(image_id,number_of_tiles))
    return img_id,tile_id,y_coord,x_coord

def apply_fetch_tiles_at_random(filepath,img_id,tile_id,y_coord,x_coord,starting_tile_id,number_of_tiles, unique_id,size = 256):
    raster = gdal.Open(filepath)
    w,h = round(raster.RasterXSize), round(raster.RasterYSize)
    
    raster = None
    try:
        img_id1,tile_id1,y_coord1,x_coord1 = fetch_tiles_at_random(0,0, w, h,size,size,starting_tile_id,number_of_tiles,unique_id)   
    except Exception as e:
        print("skipping to select tiles at random")
        print(e)
    else:
        img_id.extend(img_id1)
        tile_id.extend(tile_id1)
        y_coord.extend(y_coord1)
        x_coord.extend(x_coord1)
   
    return img_id,tile_id,y_coord,x_coord

def break_image_in_quarters(image_path,outpath,factor):
    raster = gdal.Open(image_path)
    w,h = round(raster.RasterXSize), round(raster.RasterYSize)
    bands = raster.RasterCount
    if bands>1:
        image = cv.cvtColor((utils.raster2array(raster,bands,0,0,w,h)).astype(np.uint8),cv.COLOR_RGB2BGR)
    else:
        image = (factor*utils.raster2array(raster,bands,0,0,w,h)).astype(np.uint8)
    raster = None 

    image1 = image[:h//2,:w//2,:]
    image2 = image[:h//2,w//2:,:]
    image3 = image[h//2:,:w//2,:]
    image4 = image[h//2:,w//2:,:]

    cv.imwrite(outpath.format(1), (image1).astype(np.uint8))              
    cv.imwrite(outpath.format(2), (image2).astype(np.uint8))              
    cv.imwrite(outpath.format(3), (image3).astype(np.uint8))              
    cv.imwrite(outpath.format(4), (image4).astype(np.uint8))              


def iou(y_true, y_pred):
    smooth = 1e-6
    intersection = K.sum((y_true * y_pred), axis = (0,1,2,3))
    union = K.sum((y_true + y_pred), axis = (0,1,2,3)) - K.sum((y_true * y_pred), axis = (0,1,2,3))
    
    iu = ((intersection + smooth)/ (union + smooth))
    
    return iu
    

def iou_loss(y_true, y_pred,ncl = 1.0):
    return (ncl-iou(y_true, y_pred))


 
def dice(y_true, y_pred):
    smooth1 = 1e-6
    
    num1 = K.sum((y_true *  y_pred), axis = (0,1,2,3))
    dnm1 = K.sum((y_true +  y_pred), axis = (0,1,2,3))     
    
    f1 = ((2*num1 + smooth1)/ (dnm1 + smooth1)) 
    
    return f1
 
def dice_loss(y_true, y_pred,ncl = 1.0):
    return (ncl-dice(y_true, y_pred))

def dice_bce_loss(y_true, y_pred,ncl = 1.0):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    return dice_loss(y_true, y_pred)+bce(y_true, y_pred)
