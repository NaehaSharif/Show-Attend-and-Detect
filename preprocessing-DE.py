"""Reads dicoms from "dicom folder" pre-processes them and saves them in the test folder in an npy format"""

import os
import png
import pydicom
import numpy as np
import sys
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2
plt.ion()   # interactive mode
import pydicom as dicom
import cv2
from torchvision.utils import make_grid
import random 
from PIL import Image, ImageOps
from mat4py import loadmat


##########################################################################################################################
test_dicom_folder='dicom_folder/'  # source folder 
test_npy_folder='npy_folder/'   # destination
##########################################################################################################################



def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

def crop_image(img, test=False):
    y,x = img.shape
  
    cropx=int(x*0.60) # for training anf VFA x
    
    # if test:
    #     cropx=int(x*0.60)
        

    startx = int(x*0.10)
    starty = int(y*0.50)  # for training and VFA int(y*0.50)
    return img[starty:y,startx:cropx]

def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def some_test_steps(numpy_image):
    
    
    img = Image.fromarray(np.uint16(numpy_image))
    basewidth = 300
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    return np.array(img)

            
def dicom_to_png(dicom_file, npy_file,test=False ):
    """ Function to convert from a DICOM image to png

        @param dicom_file: An opened file like object to read te dicom data
        @param npy_file: An opened file like object to write the png data
        """
  
    ans=False
    # Extracting data from the dicomfile
    
    plan = np.array(loadmat(dicom_file)['BMD']) #['LVA_SingleEnergy'])
    shape = plan.shape
    fig = plt.figure()
  

    if shape[0]:
        
    #Convert to float to avoid overflow or underflow losses.
        image_2d1 = plan.astype('uint32')
        image_2d1=np.fliplr(image_2d1) # only for new test set
        
        image_2d1=image_2d1+32767
        # cropping array
        image_2d_border=image_2d1<2**16
        image_2d_border=image_2d_border*1
        image_2d1=image_2d1*image_2d_border
        
        image_2d=crop_image(image_2d1, test)

        
            
            
        ax1 = fig.add_subplot(1,3,1)
        ax1.imshow( image_2d1, cmap=plt.cm.gray)
        ax1.set_title('Original')
        
        
    
        ax2 = fig.add_subplot(1,3,2)
        ax2.imshow(image_2d, cmap=plt.cm.gray)
        ax2.set_title('Cropped')

        image_2d = (np.maximum(image_2d,0) / image_2d1.max())
        basewidth = 300
        new_height =int( 300 * (image_2d.shape[0] / image_2d.shape[1]))
           
        image_2d=cv2.resize(image_2d, dsize=(basewidth, new_height), interpolation=cv2.INTER_AREA)
           
       
        image_2d=cv2.resize(image_2d, dsize=(300, 300), interpolation=cv2.INTER_AREA)
        ax4 = fig.add_subplot(1,3,3)
        ax4.imshow(image_2d, cmap=plt.cm.gray)
        ax4.set_title('Cropped and Resized')
        plt.tight_layout()
            

      
        np.save(npy_file, image_2d.astype(np.float32))
        ans=True

    return ans
    ###########################################################################################################



def convert_file(dicom_file_path, npy_file_path,test=False):
    """ Function to convert an dicom binary file to a
        PNG image file.

        @param dicom_file_path: Full path to the dicom file
        @param npy_file_path: Fill path to the png file
    """

    # Making sure that the dicom file exists
    if not os.path.exists(dicom_file_path):
        raise Exception('File "%s" does not exists' % dicom_file_path)

    # Making sure the png file does not exist
    if os.path.exists(npy_file_path):
        raise Exception('File "%s" already exists' % npy_file_path)

    dicom_file = open(dicom_file_path, 'rb')
    npy_file = open(npy_file_path, 'wb')

    ans=dicom_to_png(dicom_file, npy_file, test)

    npy_file.close()
    return ans


def convert_folder(dicom_folder, npy_folder, test=False):
    """ Convert all dicomfiles in a folder to png files
        in a destination folder
    """
    k_fold=1
    #Create the folder for the npy directory structure
    try:
        os.makedirs(npy_folder)
    except:
        pass

    s=partition(os.listdir(dicom_folder),k_fold)
    s.sort()
    for p,dicom_files in enumerate(s):
        for dicom_file in dicom_files:
            dicom_file_path = os.path.join(dicom_folder, dicom_file)

            # Make sure path is an actual file
            if os.path.isfile(dicom_file_path):
                npy_folder_path=npy_folder+'-'+str(p)
                if not os.path.exists(npy_folder_path):
                    os.makedirs(npy_folder_path)
                npy_file_path = os.path.join(npy_folder_path, '%s.npy' % ''.join(dicom_file.split('.')[0]))

                try:
                    # Convert the actual file
                    ans=convert_file(dicom_file_path, npy_file_path, test)
                    if ans==True:
                        print('SUCCESS: %s --> %s' % (dicom_file_path, npy_file_path))
                    else: 
                        os.remove(npy_file_path)
                except Exception as e:
                    print('FAIL: %s --> %s : %s' % (dicom_file_path, npy_file_path, e))
                    os.remove(npy_file_path)
                    
                    
                    




convert_folder(test_dicom_folder,test_npy_folder, test=True)

print('done')


