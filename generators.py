import setup
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import keras.utils
import xarray as xr
import netCDF4

from setup import *

class mockDataGenerator(keras.utils.Sequence):
    def __init__(self, num_images,sequence=1, batch_size=24, low_res_dim = 255, high_res_dim = 1068, unet = False):
        # create memory-mapped files for high_res and low_res datasets
        self.low_res_dim = low_res_dim
        self.high_res_dim = high_res_dim
        self.sequence = sequence
        self.unet = unet
        self.batch_size = batch_size
        self.num_images = num_images
        self.num_samples = self.num_images
        self.num_batches = int(np.floor(self.num_samples / self.batch_size))
        
    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        # prepare the resulting array 
        inputs = np.random.rand(self.batch_size, self.low_res_dim, self.low_res_dim, self.sequence)
        outputs = np.random.rand(self.batch_size,self.high_res_dim, self.high_res_dim)
        # random path  
        return inputs,outputs
 

class DataGeneratorMemmap(keras.utils.Sequence):
    def __init__(self, high_res_path, low_res_path, high_name, low_name, high_max, low_max, high_min, low_min, inshape = 1069, sequential=False, batch_size=24, unet = False):
        # create memory-mapped files for high_res and low_res datasets
        self.ds_nc_cerra = xr.open_dataset(high_res_path)[high_name]
        self.ds_nc_era5 = xr.open_dataset(low_res_path)[low_name]

        self.inshape = inshape
        # set pre-computed max/min for normalization 
        self.low_max = low_max 
        self.high_max = high_max
        self.low_min = low_min 
        self.high_min = high_min
        
        self.low_max = low_max 
        self.high_max = high_max
        # set boolean for sequential or random dataset
        self.sequential = sequential
        # counter for keeping track of seuquential generator 
        self.counter = 0
        # set sequence len 
        self.sequence = 1
        # flag for diffusion/unet
        self.unet = unet
        self.batch_size = batch_size
        self.num_samples = self.ds_nc_cerra.shape[0]
        self.num_batches = int(np.floor(self.num_samples / self.batch_size))
        
    def __len__(self):
        return self.num_batches

    def min_max_normalize(self, arr, max, min):
        return (arr - min) / (max - min)
    
    # must be called to restart the sequential 
    def counter_reset(self):
        self.counter = 0
    
    def __getitem__(self, idx):
        # prepare the resulting array 
        inputs = np.zeros((self.batch_size, self.inshape, self.inshape, self.sequence ))
        outputs = np.zeros((self.batch_size, 1069, 1069, 1))
        # random path 
        if(self.sequential == False):  
            #compose the batch one element at the time
            for i in range(self.batch_size):
                # get a random number in range 
                random = np.random.randint(0, (self.num_samples - self.sequence)) 
                # get the low_res items, 2 past, 1 present 1 future & normalization
                
                items = self.ds_nc_era5.isel(time=slice(random + 1,random + self.sequence + 1))
                items = items.values[:,::-1]
                
                #items = self.low_res[random + 1:random + self.sequence + 1] / self.low_max
                # swap to channel dimension 
                items = np.expand_dims(items, axis=-1)
                items = np.swapaxes(items, 0, 3)
                # upscale the images via bilinear to 256x256 
                for k in range(self.sequence):
                    inputs[i, :, :, k] = self.min_max_normalize(cv2.resize(items[0, :, :, k], (self.inshape, self.inshape), interpolation=cv2.INTER_LINEAR)
                                                           ,self.low_max,self.low_min)
                # get the target high res results 
                target = self.ds_nc_cerra.isel(time=random + self.sequence)
                target = target.values[::-1]
                #target = self.high_res[random + self.sequence]
                #normalization
                target = self.min_max_normalize(target,self.high_max,self.high_min)
                #append the target in the last place 
                outputs[i, :, :, 0] = target
        # sequential path 
        if(self.sequential == True):
            for i in range(self.batch_size):
                # get the new sequence (+1 on the last)
                items = self.ds_nc_era5.isel(time=slice(self.counter + 1,self.counter + self.sequence + 1))
                items = items.values[:,::-1]
                
                items = np.expand_dims(items, axis=-1)
                items = np.swapaxes(items, 0, 3)
                # upscale the images via bilinear to 256x256 
                for k in range(self.sequence):
                    inputs[i, :, :, k] = self.min_max_normalize(cv2.resize(items[0, :, :, k], (self.inshape, self.inshape), interpolation=cv2.INTER_LINEAR)
                                                           ,self.low_max,self.low_min)
                #get next target (+1) & normalization
                target = self.ds_nc_cerra.isel(time=self.counter + self.sequence)
                target = target.values[::-1]
                #normalization
                target = self.min_max_normalize(target,self.high_max,self.high_min)
                outputs[i,:,:,0] = target
                # update counter value 
                self.counter = self.counter + 1      
        #Diffusion takes a sequence x + y         
        if(self.unet == False):
            return np.concatenate((inputs, outputs), axis=-1)
        # The unet separates intputs and outputs
        elif(self.unet == True):
            return inputs,outputs
            
class lowDataGeneratorMemmap(keras.utils.Sequence):
    def __init__(self, high_res_path, low_res_path, high_name, low_name, high_max, low_max, high_min, low_min, sequential=False, batch_size=24):
        # create memory-mapped files for high_res and low_res datasets
        self.ds_nc_cerra = xr.open_dataset(high_res_path)[high_name]
        self.ds_nc_era5 = xr.open_dataset(low_res_path)[low_name]
        # set pre-computed max/min for normalization 
        
        self.low_max = low_max 
        self.high_max = high_max
        self.low_min = low_min 
        self.high_min = high_min
        
        # set boolean for sequential or random dataset
        self.sequential = sequential
        # counter for keeping track of seuquential generator 
        self.counter = 0
        # set sequence len 
        self.sequence = 1
        # flag for diffusion/unet
        self.batch_size = batch_size
        self.num_samples = self.ds_nc_cerra.shape[0]
        self.num_batches = int(np.floor(self.num_samples / self.batch_size))
        
    def __len__(self):
        return self.num_batches
    
    def min_max_normalize(self, arr, max, min):
        return (arr - min) / (max - min)
    
    
    # must be called to restart the sequential 
    def counter_reset(self):
        self.counter = 0
    
    def __getitem__(self, idx):
        # prepare the resulting array 
        inputs = np.zeros((self.batch_size, 256, 256, self.sequence))
        outputs = np.zeros((self.batch_size, 1069, 1069))
        # random path 
        if(self.sequential == False):  
            #compose the batch one element at the time
            for i in range(self.batch_size):
                # get a random number in range 
                random = np.random.randint(0, (self.num_samples - self.sequence)) 
                # get the low_res items, 2 past, 1 present 1 future & normalization
                
                items = self.ds_nc_era5.isel(time=slice(random + 1,random + self.sequence + 1)) 
                items = items.values[:,::-1]
                
                #items = self.low_res[random + 1:random + self.sequence + 1] / self.low_max
                # swap to channel dimension 
                items = np.expand_dims(items, axis=-1)
                items = np.swapaxes(items, 0, 3)
                # upscale the images via bilinear to 256x256 
                for k in range(self.sequence):
                    inputs[i, :, :, k] = min_max_normalize(cv2.resize(items[0, :, :, k], (256, 256), interpolation=cv2.INTER_LINEAR),low_max,low_min)

                # get the target high res results 
                target = self.ds_nc_cerra.isel(time=random + self.sequence)
                target = target.values[::-1]
                #target = self.high_res[random + self.sequence]
                #normalization
                target = self.min_max_normalize(target,self.high_max,self.high_min)
                #append the target in the last place 
                outputs[i] = target
        # sequential path 
        if(self.sequential == True):
            for i in range(self.batch_size):
                # get the new sequence (+1 on the last)
                items = self.ds_nc_era5.isel(time=slice(self.counter + 1,self.counter + self.sequence + 1)) 
                items = items.values[:,::-1]
                
                items = np.expand_dims(items, axis=-1)
                items = np.swapaxes(items, 0, 3)
                # upscale the images via bilinear to 256x256 
                for k in range(self.sequence):
                    inputs[i, :, :, k] = min_max_normalize(cv2.resize(items[0, :, :, k], (256, 256), interpolation=cv2.INTER_LINEAR),low_max,low_min)
                #get next target (+1) & normalization
                target = self.ds_nc_cerra.isel(time=self.counter + self.sequence)
                target = target.values[::-1]
                #normalization
                target = self.min_max_normalize(target,self.high_max,self.high_min)
                outputs[i] = target
                # update counter value 
                self.counter = self.counter + 1      
        # The unet separates intputs and outputs
        return inputs,outputs


class cutDataGeneratorMemmap(keras.utils.Sequence):
    def __init__(self, high_res_path, low_res_path, high_name, low_name, high_max, low_max, high_min, low_min,
                 inshape = 1069, cut_start = [100,100], sequential=False, batch_size=24, unet = False):
      
        # create memory-mapped files for high_res and low_res datasets
        self.ds_nc_cerra = xr.open_dataset(high_res_path)[high_name]
        self.ds_nc_era5 = xr.open_dataset(low_res_path)[low_name]

        self.inshape = inshape
        self.cut_start = cut_start 
        # set pre-computed max/min for normalization 
        self.low_max = low_max 
        self.high_max = high_max
        self.low_min = low_min 
        self.high_min = high_min
        # set boolean for sequential or random dataset
        self.sequential = sequential
        # counter for keeping track of seuquential generator 
        self.counter = 0
        # set sequence len 
        self.sequence = 1
        # flag for diffusion/unet
        self.unet = unet
        self.batch_size = batch_size
        self.num_samples = self.ds_nc_cerra.shape[0]
        self.num_batches = int(np.floor(self.num_samples / self.batch_size))
        
    def __len__(self):
        return self.num_batches
    
    # must be called to restart the sequential 
    def counter_reset(self):
        self.counter = 0

    def min_max_normalize(self, arr, max, min):
        return (arr - min) / (max - min)

    
    def __getitem__(self, idx):
        # prepare the resulting array 
        inputs = np.zeros((self.batch_size, 256, 256, self.sequence ))
        outputs = np.zeros((self.batch_size, 256, 256, 1))
        # random path 
        if(self.sequential == False):  
            #compose the batch one element at the time
            for i in range(self.batch_size):
                # get a random number in range 
                random = np.random.randint(0, (self.num_samples - self.sequence)) 
                # get the low_res items, 2 past, 1 present 1 future & normalization
                
                items = self.ds_nc_era5.isel(time=slice(random + 1,random + self.sequence + 1)) 
                items = items.values[:,::-1]
                #items = self.low_res[random + 1:random + self.sequence + 1] / self.low_max
                # swap to channel dimension 
                items = np.expand_dims(items, axis=-1)
                items = np.swapaxes(items, 0, 3)

                for k in range(self.sequence):
                    tmp = cv2.resize(items[0, :, :, k], (self.inshape, self.inshape), interpolation=cv2.INTER_LINEAR)
                    inputs[i, :, :, k] = self.min_max_normalize(tmp[self.cut_start[0]:self.cut_start[0]+256,self.cut_start[1]:self.cut_start[1]+256]
                                                           ,self.low_max,self.low_min)
                # get the target high res results 
                target = self.ds_nc_cerra.isel(time=random + self.sequence)
                target = target.values[::-1]
                target = target[self.cut_start[0]:self.cut_start[0]+256,self.cut_start[1]:self.cut_start[1]+256]
                target = self.min_max_normalize(target,self.high_max,self.high_min)
                #target = self.high_res[random + self.sequence]
                #normalization
                #append the target in the last place 
                outputs[i, :, :, 0] = target
        # sequential path 
        if(self.sequential == True):
            for i in range(self.batch_size):
                # get the new sequence (+1 on the last)
                items = self.ds_nc_era5.isel(time=slice(self.counter + 1,self.counter + self.sequence + 1)) / self.low_max
                items = items.values[:,::-1]
                
                items = np.expand_dims(items, axis=-1)
                items = np.swapaxes(items, 0, 3)
                # upscale the images via bilinear to 256x256 
                for k in range(self.sequence):
                    tmp = cv2.resize(items[0, :, :, k], (self.inshape, self.inshape), interpolation=cv2.INTER_LINEAR)
                    inputs[i, :, :, k] = self.min_max_normalize(tmp[self.cut_start[0]:self.cut_start[0]+256,self.cut_start[1]:self.cut_start[1]+256]
                                                           ,self.low_max,self.low_min)
                #get next target (+1) & normalization
                target = self.ds_nc_cerra.isel(time=self.counter + self.sequence)
                target = target.values[::-1]
                target = target[self.cut_start[0]:self.cut_start[0]+256,self.cut_start[1]:self.cut_start[1]+256]
                target = self.min_max_normalize(target,self.high_max,self.high_min)
                #normalization
                outputs[i,:,:,0] = target
                # update counter value 
                self.counter = self.counter + 1      
        #Diffusion takes a sequence x + y         
        if(self.unet == False):
            return np.concatenate((inputs, outputs), axis=-1)
        # The unet separates intputs and outputs
        elif(self.unet == True):
            return inputs,outputs
        

