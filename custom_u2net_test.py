import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB


def test_model(model):

    
    # ------- 1. set the directory of test dataset --------

    test_data_dir = os.path.join(os.getcwd(), 'my_data' + os.sep)
    test_image_dir = os.path.join('TDP_test_dataset','TDP_IMAGES' + os.sep)
    test_label_dir = os.path.join('TDP_test_dataset','TDP_MASKS' + os.sep)
   

   
    image_ext = '.jpg'
    label_ext = '.png'

    batch_size_val = 1

    test_img_name_list = glob.glob(test_data_dir + test_image_dir + '*' + image_ext)

    test_lbl_name_list = []
    for img_path in test_img_name_list:
        img_name = img_path.split(os.sep)[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1,len(bbb)):
            imidx = imidx + "." + bbb[i]

        test_lbl_name_list.append(test_data_dir + test_label_dir + imidx + label_ext)

    test_salobj_dataset = SalObjDataset(
        img_name_list=test_img_name_list,
        lbl_name_list=test_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=batch_size_val,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 2. test process ---------
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    total_pixels = 0
    correct_pixels = 0
    accuracy = 0

    with torch.no_grad():
        for i, data in enumerate(test_salobj_dataloader):
            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            predicted_masks = (outputs[0] > 0.5).float() 

            total_pixels = labels.numel()
            correct_pixels = (predicted_masks == labels).sum().item()
            accuracy += (correct_pixels/total_pixels)*100

        avr_accuracy=accuracy/len(test_salobj_dataloader)   
        print(f'avarage_accuracy: {avr_accuracy}%')


if __name__ == "__main__":
        pass