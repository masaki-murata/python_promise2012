#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 19:00:30 2018

@author: muratamasaki
"""

import readmhd
import os, re, csv, sys
import attention_senet_based
from skimage.transform import resize
import numpy as np
from seunet_model import seunet
import keras

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
    
def crop_3d(crop_shape=np.array([32,256,256]),
            ):
    path_to_mri = "../IntermediateData/rescaled/Case%02d.npy"
    path_to_segmentation = "../IntermediateData/rescaled/Case%02d_segmentation.npy"
    path_to_save_dir = "../IntermediateData/cropped/"
    path_to_mri_cropped = path_to_save_dir + "Case%02d.npy"
    path_to_segmentation_cropped = path_to_save_dir + "Case%02d_segmentation.npy"
    
    if not os.path.isdir(path_to_save_dir):
        os.mkdir(path_to_save_dir)
    for case in range(50):
        mri = np.load(path_to_mri % case)
        segmentation = np.load(path_to_segmentation % case)
        if mri.shape != segmentation.shape:
            print("shape does not match!")
            sys.exit()
        mri_shape = np.array(mri.shape)
        min_pos = (mri_shape-crop_shape)//2
        max_pos = min_pos + crop_shape
        mri_cropped = mri[min_pos[0]:max_pos[0], min_pos[1]:max_pos[1], min_pos[2]:max_pos[2]]
        segmentation_cropped = segmentation[min_pos[0]:max_pos[0], min_pos[1]:max_pos[1], min_pos[2]:max_pos[2]]
        np.save(path_to_mri_cropped % case, mri_cropped)
        np.save(path_to_segmentation_cropped % case, segmentation_cropped)


def make_data_label(crop_shape=np.array([32,256,256])):
    path_to_mri = "../IntermediateData/cropped/Case%02d.npy"
    path_to_segmentation = "../IntermediateData/cropped/Case%02d_segmentation.npy"
    path_to_image = "../IntermediateData/data_for_train/data.npy"
    path_to_target = "../IntermediateData/data_for_train/label.npy"
    path_to_data_dir = "../IntermediateData/data_for_train/"
    if not os.path.isdir(path_to_data_dir):
        os.mkdir(path_to_data_dir)
    
    data = np.zeros((50,)+tuple(crop_shape)+(1,), dtype=np.float)
    label = np.zeros((50,)+tuple(crop_shape)+(1,), dtype=np.int8)
    for case in range(50):
        mri = np.load(path_to_mri % case)
        segmentation = np.load(path_to_segmentation % case)
        data[case] = mri.reshape(mri.shape+(1,))
        label[case] = segmentation.reshape(segmentation.shape+(1,))
    np.save(path_to_image, data)
    np.save(path_to_target, label)
    

def predict(path_to_image, path_to_target, model_path, batch_size, path_to_output):
    # X_sketchが元の病理画像。[0,1]に規格化されたnp array
    X_sketch_train = np.load(path_to_image)
    # X_fullがsegmentationされた画像。[0,1]に規格化された4channel np array
    X_full_train = np.load(path_to_target)


    img_dim = X_full_train.shape[-4:]
    img_dim0 = X_sketch_train.shape[-4:]
    train_size = X_full_train.shape[0]

    if img_dim[:-1] != img_dim0[:-1]:
        print("Error: output shape must be the same as that of input")

#    opt_generator = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model = seunet(img_dim0, img_dim)
    
    model.load_weights(model_path)
    
    Y_train = model.predict(X_sketch_train, batch_size)
    
    np.save(path_to_output, Y_train.reshape(Y_train.shape[:-1]))

#resize_all(ref_size=voxel_size)
#crop_3d()
#make_data_label()

path_to_image = "../IntermediateData/data_for_train/data.npy"
path_to_target = "../IntermediateData/data_for_train/label.npy"
model_path = "../IntermediateData/model/seunet_weights_30.h5"
batch_size = 1
path_to_output = "../IntermediateData/output/segmentation_prediction.npy"
predict(path_to_image, path_to_target, model_path, batch_size, path_to_output)