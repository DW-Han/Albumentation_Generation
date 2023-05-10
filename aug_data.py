import os
import random
import cv2
import numpy as np
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
import ipdb 
import pickle as pkl
import imageio
import torch
import torchvision.transforms as transforms
import torchvision.utils as utils

# Set the paths to the image and mask directories
IMG_DIR = '/data/new_iPhone_Seg_Dataset_removal/new_iPhone_Seg_Dataset/test/images/Unit_03'
MASK_DIR = '/data/new_iPhone_Seg_Dataset_removal/new_iPhone_Seg_Dataset/test/masks/Unit_03'

SAVE_DIR = '/data/augemented_set2/'

def apply_transformations(img_path, masks_list):
    #ipdb.set_trace()
    img = cv2.imread(img_path)
    masks = masks_list

    # Define the transformation pipeline
    transform = A.Compose(
        [
            A.SafeRotate(limit=30, p=0.5),
            #A.RandomScale(scale_limit=0.1, interpolation=1, p=0.5),
            #A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    for i in range(100):
        # Apply the transformation to the image and mask
        transformed = transform(image=img, masks=masks)
        transformed_img = transformed['image']
        transformed_masks = transformed['masks']

        grayscale_transform = transforms.Grayscale()
        gray_img = grayscale_transform(transformed_img)
        #ipdb.set_trace()
        utils.save_image(transformed_img, os.path.join(SAVE_DIR,"images", os.path.basename(IMG_DIR),str(i)+'_' + os.path.basename(IMG_DIR) +'.jpg'))

        os.makedirs(os.path.join(SAVE_DIR,"masks",os.path.basename(MASK_DIR), str(i)+'_' + os.path.basename(IMG_DIR)), exist_ok=True)
        for count, j in enumerate(transformed_masks):
            print(count)
            if count == 0:
                np.save(os.path.join(SAVE_DIR,"masks",os.path.basename(MASK_DIR), str(i)+'_' + os.path.basename(IMG_DIR),'battery.npy'), j)
                print("here")
            elif count == 1:
                np.save(os.path.join(SAVE_DIR,"masks",os.path.basename(MASK_DIR), str(i)+'_' + os.path.basename(IMG_DIR),'screw.npy'), j)
        #ipdb.set_trace()


   
os.makedirs(os.path.join(SAVE_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'images', os.path.basename(IMG_DIR)), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'masks'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'masks', os.path.basename(MASK_DIR)), exist_ok=True)

for img_name in os.listdir(IMG_DIR):

    if img_name.endswith('.jpg') or img_name.endswith('.png'):
        mask_name = img_name.replace('.jpg', '')
        img_path = os.path.join(IMG_DIR, img_name)

        battery_path = os.path.join(MASK_DIR, mask_name, "battery.npy")
        screw_path = os.path.join(MASK_DIR, mask_name, "screw.npy")

        #print(screw_path)

        if os.path.exists(battery_path) and os.path.exists(screw_path):
            #print("here0")
            apply_transformations(img_path,[np.load(battery_path, allow_pickle=True),np.load(screw_path, allow_pickle=True)])