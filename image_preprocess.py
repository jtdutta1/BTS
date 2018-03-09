# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 12:51:49 2018
@version 0.2.1
@author: jtdut
"""

""" An image preprocessor """
import SimpleITK as sitk
import numpy as np


class imagepreprocess():
    """ This class does some preprocessing for the image. These includes extracting different views
    from the .mha file of a patient, returning the 3 different views of the MRI scans, create channels 
    for passing to a CNN and also a normalization function."""
    
    def __init__(self,image_array = None, path = None):
        if(image_array == None and not path == None):
            self._img = sitk.ReadImage(path)
        elif(path == None and not image_array == None):
            self._img = sitk.GetImageFromArray(image_array)
        else:
            raise Exception('Either a 2D image array has to be provided or a path to the mha file has to be set. Both cannot be None')
        '''self._img_top = self.__getTopView()
        self._img_side = self.__getSideView()
        self._img_back = self.__getBackView()'''
        
    def _getTopView(self,slices = None):
        """ Returns the volumetric MRI scan of the brain from the top.
        
        INPUT:-
        slices(Optional): None or list. Accepts a list of 2 elements and returns the MRI scans between 
                        these 2 ranges
                        
        OUTPUT:-
        A 3-D numpy array with the volumetric top-side MRI scans of the brain.
        """
        if(slices == None or len(slices) == 0):
            return sitk.GetArrayFromImage(self._img)
        else:
            if(len(slices)>2):
                raise Exception('Warning!! There are more than 2 elements and we are considering the first 2')
                return sitk.GetArrayFromImage(self._img)[slices[0]:slices[1]]
            elif(len(slices)<2):
                raise Exception('slices needs 2 parameters to return the custom MRI scan slices.')
            else:
                return sitk.GetArrayFromImage(self._img)[slices[0]:slices[1]]
    def _getSideView(self,slices = None):
        """ Returns the volumetric MRI scan of the brain from the side.
        
        INPUT:-
        slices(Optional): None or list. Accepts a list of 2 elements and returns the MRI scans between 
                        these 2 ranges
                        
        OUTPUT:-
        A 3-D numpy array with the volumetric side MRI scans of the brain.
        """
        img = []
        for e in range(-239,1):
            img.append(sitk.GetArrayFromImage(self._img)[:,:,e])
        img = np.array(img)
        if(slices == None or len(slices) == 0):
            return img
        else:
            if(len(slices)>2):
                raise Exception('Warning!! There are more than 2 elements and we are considering the first 2')
                print
                return img[slices[0]:slices[1]]
            elif(len(slices)<2):
                raise Exception('slices needs 2 parameters to return the custom MRI scan slices.')
            else:
                return img[slices[0]:slices[1]]
    
    def _getBackView(self,slices = None):
        """ Returns the volumetric MRI scan of the brain from the back.
        
        INPUT:-
        slices(Optional): None or list. Accepts a list of 2 elements and returns the MRI scans between 
                        these 2 ranges
                        
        OUTPUT:-
        A 3-D numpy array with the volumetric backside MRI scans of the brain.
        """
        img = []
        for e in range(-239,1):
            img.append(sitk.GetArrayFromImage(self._img)[:,e,:])
        img = np.array(img)
        if(slices == None or len(slices) == 0):
            return img
        else:
            if(len(slices)>2):
                raise Exception('Warning!! There are more than 2 elements and we are considering the first 2')
                return img[slices[0]:slices[1]]
            elif(len(slices)<2):
                raise Exception('slices needs 2 parameters to return the custom MRI scan slices.')
            else:
                return img[slices[0]:slices[1]]
                
    def normalize(self,image_array):
        """ Reads and returns an image consisting of normalized values.
        
        INPUT:-
        
        image_array: A 2-D or 3-D numpy array representing the pixel values.
        
        OUTPUT:-
        
        The numpy array of the same shape with each value normalized between 0 and 1.
        """
        arr_max = np.max(image_array); arr_min = np.min(image_array)
        l = []
        for e in image_array:
            l1 = []
            for i in e:
                l1.append((i-arr_min)/(arr_max-arr_min))
            l.append(l1)
        return np.array(l)
    
    def create_channel(self,image_array):
        """ Reads and returns an image consisting of a single color channel.
        NOTE: Single color channel cannot be displayed using matplotlib.pyplot.imshow() method.
        
        INPUT:-
        image_array: A 2-D or 3-D numpy array representing the pixel values.
        
        OUTPUT:-
        The equivalent numpy array with a higher(1 extra) dimension.
        """
        dim = len(image_array.shape)
        if(dim == 2):
            l = []
            for e in image_array:
                l1 = []
                for i in e:
                    l1.append([i])
                l.append(l1)
            return np.array(l)
        elif(dim == 3):
            l = []
            for i in image_array:
                l1 = []
                for j in i:
                    l2 = []
                    for k in j:
                        l2.append([k])
                    l1.append(l2)
                l.append(l1)
            return np.array(l)
        else:
            raise Exception('Expected 2 or 3 dimensions. Got '+dim+'.')
        
    
    def __future_works(self):
        return