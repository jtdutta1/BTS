# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 12:51:49 2018
@version 0.1.3
@author: jtdut
"""

''' An image preprocessor '''
import SimpleITK as sitk
import numpy as np


class imagepreprocess():
    # Object accepts the path of the MHA file to be extracted
    def __init__(self,image_array = None, path = None):
        if(image_array == None and not path == None):
            self.__img = sitk.ReadImage(path)
        elif(path == None and not image_array == None):
            self.__img = sitk.GetImageFromArray(image_array)
        else:
            raise Exception('Either a 2D image array has to be provided or a path to the mha file has to be set. Both cannot be None')
        self.__img_top = self.__getTopView()
        self.__img_side = self.__getSideView()
        self.__img_back = self.__getBackView()
        
    def __getTopView(self,slices = None):
        ''' Returns a 3D numpy array of the top view
        
        slices accepts a list of 2 integers.
        The first number indicates the starting position and the second one indicates the last position.
        '''
        if(slices == None or len(slices) == 0):
            return sitk.GetArrayFromImage(self.__img)
        else:
            if(len(slices)>2):
                raise Exception('Warning!! There are more than 2 elements and we are considering the first 2')
                return sitk.GetArrayFromImage(self.__img)[slices[0]:slices[1]]
            elif(len(slices)<2):
                raise Exception('slices needs 2 parameters to return the custom MRI scan slices.')
    def __getSideView(self,slices = None):
        ''' slices accepts a list of 2 integers.
            The first number indicates the starting position and the second one indicates the last position.
        '''
        img = []
        for e in range(-239,1):
            img.append(sitk.GetArrayFromImage(self.__img)[:,:,e])
        img = np.array(img)
        if(slices == None or len(slices) == 0):
            return img
        else:
            if(len(slices)>2):
                raise Exception('Warning!! There are more than 2 elements and we are considering the first 2')
                return img[slices[0]:slices[1]]
            elif(len(slices)<2):
                raise Exception('slices needs 2 parameters to return the custom MRI scan slices.')
    
    def __getBackView(self,slices = None):
        ''' slices accepts a list of 2 integers.
            The first number indicates the starting position and the second one indicates the last position.
        '''
        img = []
        for e in range(-239,1):
            img.append(sitk.GetArrayFromImage(self.__img)[:,e,:])
        img = np.array(img)
        if(slices == None or len(slices) == 0):
            return img
        else:
            if(len(slices)>2):
                raise Exception('Warning!! There are more than 2 elements and we are considering the first 2')
                return img[slices[0]:slices[1]]
            elif(len(slices)<2):
                raise Exception('slices needs 2 parameters to return the custom MRI scan slices.')
                
    def normalize(self,image_array):
        '''Accepts a 2D numpy array and returns another 2D numpy array of the same size with normalized values.
        '''
        arr_max = np.max(image_array); arr_min = np.min(image_array)
        l = []
        for e in image_array:
            l1 = []
            for i in e:
                l1.append((i-arr_min)/(arr_max-arr_min))
            l.append(l1)
        return np.array(l)
    
    def create_channel(self,image_array):
        ''' Accepts a 2D numpy array and returns a 3D numpy array
        '''
        l = []
        for e in image_array:
            l1 = []
            for i in e:
                l1.append([i])
            l.append(l1)
        return np.array(l)
    
    def __future_works(self):
        return