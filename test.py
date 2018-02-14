#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 19:00:30 2018

@author: muratamasaki
"""

import readmhd
import os, re, csv
import attention_senet_based
from skimage.transform import resize
import numpy as np

#model = attention_senet_based.seunet(img_dims=(30,30,30,1))
#model.summary()
#
path_to_dir = "../TrainingData_Part%d/"
path_to_voxelsize = "../IntermediateData/voxelsize.csv"
#voxelsize = open(path_to_voxelsize, 'w')
#writer = csv.writer(voxelsize, lineterminator='\n') 
#writer.writerow(["Case","voxelsize", "physical_size"]) # headder

physical_sizes = np.zeros((50,3))
voxel_sizes = np.zeros((50,3))
count=0
for part in range(1,4):
    for file in os.listdir(path_to_dir % part):
        if re.match("Case[0-9][0-9].mhd", file):
            volume = readmhd.read(path_to_dir % part + file)
            voxel_sizes[count] = volume.voxelsize[::-1]
#            physical_size = np.array(volume.voxelsize)*np.array(volume.matrixsize)
#            writer.writerow([file, volume.voxelsize[::-1], physical_size[::-1]])
            physical_sizes[count] = (np.array(volume.voxelsize)*np.array(volume.matrixsize))[::-1]
            count += 1
z_min, z_argmin = np.min(physical_sizes[:,0]), np.argmin(physical_sizes[:,0])
y_min, y_argmin  = np.min(physical_sizes[:,1]), np.argmin(physical_sizes[:,1])
x_min, x_argmin  = np.min(physical_sizes[:,2]), np.argmin(physical_sizes[:,2])
print("z_min={0}, z_argmin={1}, z_ref_vox={2}".format(z_min, z_argmin, voxel_sizes[z_argmin, 0]))
print("y_min={0}, y_argmin={1}, y_ref_vox={2}".format(y_min, y_argmin, voxel_sizes[y_argmin, 1]))
print("x_min={0}, x_argmin={1}, x_ref_vox={2}".format(x_min, x_argmin, voxel_sizes[x_argmin, 2]))

voxel_size = np.array([z_min/32, y_min/256, x_min/256])
#voxelsize.close()

def resize_all(ref_size=np.array([3.6, 0.625, 0.625])):
    for part in range(1,4):
        for file in os.listdir(path_to_dir % part):
            if re.match("Case[0-9][0-9].mhd", file):
                volume = readmhd.read(path_to_dir % part + file)
                matrixsize_rescaled = (volume.matrixsize[::-1]*(volume.voxelsize[::-1] / ref_size)).astype(np.int)
                matrixsize_rescaled[1]=matrixsize_rescaled[2]
                volume_rescaled = resize(volume.vol, matrixsize_rescaled)
#                np.float32(volume_rescaled).tofile("../IntermediateData/rescaled/"+file[:-4]+".raw")
                np.save("../IntermediateData/rescaled/"+file[:-4]+".npy", np.float32(volume_rescaled))
                print(file, volume_rescaled.shape)
            if re.match("Case[0-9][0-9]_segmentation.mhd", file):
                volume = readmhd.read(path_to_dir % part + file)
                matrixsize_rescaled = (volume.matrixsize[::-1]*(volume.voxelsize[::-1] / ref_size)).astype(np.int)
                matrixsize_rescaled[1]=matrixsize_rescaled[2]
                volume_rescaled = resize(np.float32(volume.vol), matrixsize_rescaled)
#                volume_rescaled.tofile("../IntermediateData/rescaled/"+file[:-4]+".raw")
                np.save("../IntermediateData/rescaled/"+file[:-4]+".npy", np.int8(volume_rescaled))
                print(file, volume_rescaled.shape)
    
resize_all(ref_size=voxel_size)