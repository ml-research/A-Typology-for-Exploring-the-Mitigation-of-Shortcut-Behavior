#!/usr/bin/env python3
# original from https://github.com/laura-rieger/deep-explanation-penalization/tree/master/isic-skin-cancer
""""Utils for ISIC19 dataset setup."""
import os
from os.path import join as oj
import csv
import time

import numpy as np
from numpy.lib.nanfunctions import _nancumprod_dispatcher
from skimage.morphology import dilation
from skimage.morphology import square
from PIL import Image
from tqdm import tqdm 
import matplotlib.pyplot as plt
import h5py

from isic_api import ISICApi


def download_images(data_path, num_imgs):
    api = ISICApi()
    savePath = os.path.join(data_path, 'raw')
    #print()
    #print(savePath)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    start_offset = 0

    print(F"Start downloading {num_imgs} images from ISIC API. This could take some time ...")
    for i in tqdm(range(int(num_imgs/50)+1)):
    #for i in range(1):    
        imageList = api.getJson('image?limit=50&offset=' + str(start_offset) + '&sort=name')
        for image in imageList:
            #print(image['_id'])
            imageFileResp = api.get('image/%s/download' % image['_id'])
            imageFileResp.raise_for_status()
            imageFileOutputPath = os.path.join(savePath, '%s.jpg' % image['name'])
            with open(imageFileOutputPath, 'wb') as imageFileOutputStream:
                for chunk in imageFileResp:
                    imageFileOutputStream.write(chunk)
        start_offset +=50

def download_seg_masks(data_path, num_imgs):
    api = ISICApi()
    savePath = os.path.join(data_path, 'segmentation_hint_raw')
    #print()
    #print(savePath)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    start_offset = 0

    print(F"Start downloading {num_imgs} images segmentation masks from ISIC API. This could take some time ...")
    for i in tqdm(range(int(num_imgs/50)+1)):
    #for i in range(1):    
        imageList = api.getJson('image?limit=50&offset=' + str(start_offset) + '&sort=name')
        for image in imageList:
            #print(image['_id']) segmentation/
            imageSegResp = api.getJson('segmentation?imageId=%s' % image['_id'])
            try:
                imageSegResp2 = api.get('segmentation/%s/mask' % imageSegResp[0]['_id'])
                imageSegResp2.raise_for_status()
                imageFileOutputPath = os.path.join(savePath, '%s.png' % image['name'])
                with open(imageFileOutputPath, 'wb') as imageFileOutputStream:
                    for chunk in imageSegResp2:
                        imageFileOutputStream.write(chunk)
            except:
                print(f"Skipping image {image['_id']} - name={image['name']}. No seg mask!")
        start_offset +=50

def download_metadata(data_path, num_imgs):
    api = ISICApi()
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    imageList = api.getJson('image?limit=' + str(num_imgs) +'&offset=0&sort=name')
        
    print('Fetching metadata for %s images...' % len(imageList))
    imageDetails = []
    for image in tqdm(imageList):
    
        # Fetch the full image details
        imageDetail = api.getJson('image/%s' % image['_id'])
        imageDetails.append(imageDetail)

    # Determine the union of all image metadata fields
    metadataFields = set(
            field
            for imageDetail in imageDetails
            for field in imageDetail['meta']['clinical'].keys()
        )

    metadataFields = ['isic_id'] + sorted(metadataFields)
    outputFileName = "meta"

    outputFilePath = os.path.join(data_path, outputFileName)
    # Write the metadata to a CSV
    print('Writing metadata to CSV: %s' % outputFileName+'.csv')
    with open(outputFilePath+'.csv', 'w') as outputStream:
        csvWriter = csv.DictWriter(outputStream, metadataFields)
        csvWriter.writeheader()
        for imageDetail in imageDetails:
            rowDict = imageDetail['meta']['clinical'].copy()
            rowDict['isic_id'] = imageDetail['name']
            csvWriter.writerow(rowDict)

def sort_images(data_path):
    img_path = os.path.join(data_path, "raw")
    processed_path = os.path.join(data_path, "processed")
    #segmentation_path = os.path.join(data_path, "segmentation")
    benign_path = os.path.join(processed_path, "no_cancer")
    malignant_path = os.path.join(processed_path, "cancer")
    os.makedirs(processed_path,exist_ok = True)
    os.makedirs(benign_path,exist_ok = True)
    #os.makedirs(segmentation_path,exist_ok = True)
    os.makedirs(malignant_path,exist_ok = True)

    list_of_meta = []
    print("Sort images by benign and malgignant...")
    with open(oj(data_path, "meta.csv"), newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        next(spamreader)
        for row in spamreader:
            list_of_meta.append(row)    

    list_benign_files = []
    for line in list_of_meta[:]:
        if len(line) > 0 and line[3] == 'benign':
            list_benign_files.append(line[0] + ".jpg")
    list_mal_files = []
    for line in list_of_meta[:]:
        if len(line) > 0 and line[3] == 'malignant':
            list_mal_files.append(line[0] + ".jpg")

    def resize_and_save(my_list, my_folder):
        for i,file_name in tqdm(enumerate(my_list)):
            try:
                img = Image.open(oj(img_path, file_name))
                test = np.asarray(img)
                #test_new = imresize(test, (299, 299, 3))
                resized_img = Image.fromarray(test).resize(size=(299, 299))
                resized_img.save(oj(my_folder, file_name))
                #scipy.misc.imsave(oj(my_folder, file_name), resized_img)
            except:
                print(file_name)
    print(f"Resize and save {len(list_mal_files)} cancer images...")
    resize_and_save(list_mal_files, malignant_path)
    print(f"Resize and save {len(list_benign_files)} not-cancer images...")
    resize_and_save(list_benign_files, benign_path)

def resize_and_save_seg_masks_hint(data_path):
    print("Preprocess segmenatation hint masks...")
    seg_path = os.path.join(data_path, "segmentation_hint_raw")
    processed_path = os.path.join(data_path, "segmentation_hint")
    os.makedirs(processed_path,exist_ok = True)

    for i, file_name in tqdm(enumerate(os.listdir(seg_path))):
        try:
            img = Image.open(oj(seg_path, file_name))
            test = np.asarray(img)
            #test_new = imresize(test, (299, 299, 3))
            resized_img = Image.fromarray(test).resize(size=(299, 299))
            resized_img.save(oj(processed_path, file_name))
            #scipy.misc.imsave(oj(my_folder, file_name), resized_img)
        except:
            print(file_name)

def save_isic_2019_to_h5_file(data_path, num_imgs, only_hint=False):
    if not os.path.exists(data_path + "/segmentation"):
        raise Exception(f"No segmentation at {data_path} provided! Download from https://github.com/laura-rieger/deep-explanation-penalization")

    processed_path = os.path.join(data_path, "processed")
    benign_path = os.path.join(processed_path, "no_cancer")
    malignant_path = os.path.join(processed_path, "cancer")
    #feature_path = os.path.join(data_path, "calculated_features")
    segmentation_path = os.path.join(data_path, "segmentation")
    segmentation_hint_path = os.path.join(data_path, "segmentation_hint")

    if not os.path.exists(segmentation_hint_path):
        print(f"No segmentation hint masks at {data_path} provided!")
        print("Download Hint masks..")
        download_seg_masks(data_path=data_path, num_imgs=num_imgs)
        resize_and_save_seg_masks_hint(data_path)

    # used for converting to the range VGG16 is used to
    mean = np.asarray([0.485, 0.456, 0.406])
    std = np.asarray([0.229, 0.224, 0.225])

    list_of_img_names = os.listdir(benign_path)
    # __import__("pdb").set_trace()
    # store no_cancer images
    imgages_np = np.empty((len(list_of_img_names), 3, 299, 299))
    masks_np = np.zeros((len(list_of_img_names), 1, 299, 299))
    masks_hint_np = np.zeros((len(list_of_img_names), 1, 299, 299))
    flags_np = np.zeros(len(list_of_img_names)).astype(np.int64)
    flags_hint_np = np.zeros(len(list_of_img_names)).astype(np.int64)

    my_square = square(20)
    print("Load and preprocess not_cancer images and segmentation masks...")
    for i in tqdm(range(len(list_of_img_names))):
        img = Image.open(oj(benign_path, list_of_img_names[i]))
        # normalize 
        img_np = ((np.asarray(img)/255.0 -mean)/std).swapaxes(0,2).swapaxes(1,2)
        img.close()
        imgages_np[i] = img_np
        # check if a segmentation feedback mask exits
        if os.path.isfile(oj(segmentation_path, list_of_img_names[i])):
            seg = Image.open(oj(segmentation_path, list_of_img_names[i]))
            seg_np =  dilation((np.asarray(seg)[:,:, 0] > 100).astype(np.uint8),my_square).astype(np.float32)
            #show_image_numpy(seg_np, shape=(299,299))
            masks_np[i] = np.expand_dims(seg_np, axis=0)
            flags_np[i] = 1
        # check if seg masks hint exist --> .png insted of .jpg
        png_name = list_of_img_names[i][:-3] + 'png'
        if os.path.isfile(oj(segmentation_hint_path, png_name)):
            seg_hint = Image.open(oj(segmentation_hint_path, png_name))
            seg_hint_np = (np.asarray(seg_hint)/255.0).astype(np.float32)
            filter = seg_hint_np > 0.
            seg_hint_np[filter] = 1.0
            #util.show_image_numpy(seg_hint_np, shape=(299,299))
            masks_hint_np[i] = np.expand_dims(seg_hint_np, axis=0)
            flags_hint_np[i] = 1

    print("Save and compress ISIC 2019 cancer to .h5 file...")
    # start = time.time()

    if only_hint:
        print(f"Not cancer -> Save only hint masks and flags!")
        with h5py.File(oj(data_path + 'not_cancer_masks_hint.h5'), 'w') as hf:
                hf.create_dataset("not_cancer_masks_hint",  data=masks_hint_np)
        with h5py.File(oj(data_path + 'not_cancer_flags_hint.h5'), 'w') as hf:
                hf.create_dataset("not_cancer_flags_hint",  data=flags_hint_np)
    else:
        with h5py.File(oj(data_path + 'not_cancer_imgs.h5'), 'w') as hf:
            hf.create_dataset("not_cancer_imgs",  data=imgages_np)
        with h5py.File(oj(data_path + 'not_cancer_masks.h5'), 'w') as hf:
            hf.create_dataset("not_cancer_masks",  data=masks_np)
        with h5py.File(oj(data_path + 'not_cancer_flags.h5'), 'w') as hf:
            hf.create_dataset("not_cancer_flags",  data=flags_np)
        with h5py.File(oj(data_path + 'not_cancer_masks_hint.h5'), 'w') as hf:
                hf.create_dataset("not_cancer_masks_hint",  data=masks_hint_np)
        with h5py.File(oj(data_path + 'not_cancer_flags_hint.h5'), 'w') as hf:
                hf.create_dataset("not_cancer_flags_hint",  data=flags_hint_np)

    del masks_hint_np
    del flags_hint_np
    del flags_np
    del masks_np
    del imgages_np
    
    list_of_img_names = os.listdir(malignant_path)
    imgages_cancer_np = np.empty((len(list_of_img_names), 3, 299, 299))
    masks_hint_np = np.zeros((len(list_of_img_names), 1, 299, 299))
    flags_hint_np = np.zeros(len(list_of_img_names)).astype(np.int64)

    print("Load and preprocess cancer images...")
    index_to_delete = []
    for i in tqdm(range(len(list_of_img_names))):
        try:
            img = Image.open(oj(malignant_path, list_of_img_names[i]))
            img_np = ((np.asarray(img)/255.0 -mean)/std).swapaxes(0,2).swapaxes(1,2)
            img.close()
            imgages_cancer_np[i] = img_np

            png_name = list_of_img_names[i][:-3] + 'png'
            if os.path.isfile(oj(segmentation_hint_path, png_name)):
                seg_hint = Image.open(oj(segmentation_hint_path, png_name))
                seg_hint_np = (np.asarray(seg_hint)/255.0).astype(np.float32)
                filter = seg_hint_np > 0.
                seg_hint_np[filter] = 1.0
                #util.show_image_numpy(seg_hint_np, shape=(299,299))
                masks_hint_np[i] = np.expand_dims(seg_hint_np, axis=0)
                flags_hint_np[i] = 1
        except:
            print(f"Img {list_of_img_names[i]} at index {i} cannot be converted!")
            index_to_delete.append(i)

    imgages_cancer_np = np.delete(imgages_cancer_np, index_to_delete, axis=0)
    masks_hint_np = np.delete(masks_hint_np, index_to_delete, axis=0)
    flags_hint_np = np.delete(flags_hint_np, index_to_delete, axis=0)

   
    if only_hint:
        print(f"Cancer -> Save only hint masks and flags!")
   
        with h5py.File(oj(data_path + 'cancer_masks_hint.h5'), 'w') as hf:
            hf.create_dataset("cancer_masks_hint",  data=masks_hint_np)
        with h5py.File(oj(data_path + 'cancer_flags_hint.h5'), 'w') as hf:
            hf.create_dataset("cancer_flags_hint",  data=flags_hint_np)
    else:
        with h5py.File(oj(data_path + 'cancer_imgs.h5'), 'w') as hf:
            hf.create_dataset("cancer_imgs",  data=imgages_cancer_np)
        with h5py.File(oj(data_path + 'cancer_masks_hint.h5'), 'w') as hf:
            hf.create_dataset("cancer_masks_hint",  data=masks_hint_np)
        with h5py.File(oj(data_path + 'cancer_flags_hint.h5'), 'w') as hf:
            hf.create_dataset("cancer_flags_hint",  data=flags_hint_np)


    print("Save and compress ISIC 2019 cancer to .h5 file...")


def isic_setup(data_path="ISIC19/", num_imgs=25000):
    if not os.path.exists(data_path):
        print("Create ISIC19 folder ...")
        os.makedirs(data_path)

    if not os.path.exists(data_path + "/raw"):
        print("Download raw files via api...")
        download_images(data_path, num_imgs)
        download_metadata(data_path, num_imgs)
        print("Download raw images and metadata finished!")
    else:
        print("Raw images files already exist!!")
    
    if not os.path.exists(data_path + "/processed"):
        print("Sort benign/malignant files ...")
        sort_images(data_path)
    else:
        print("Processed files already exist!!")

    sort_images(data_path)
    save_isic_2019_to_h5_file(data_path="ISIC19/", num_imgs=num_imgs)

# TO SETUP THE ISIC19 DATASET DO THE FOLLOWING:
# # cd to ISIC_2019 folder and run from terminal command: python3 setup_isic.py
num_imgs = 25000 #100
isic_setup(num_imgs=num_imgs)
