import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

import onnx


def main():

    # --------- 1. get image path and name ---------
    image_dir = './test_data/test_portrait_images/portrait_im/img_1585.png'

    model_dir = './saved_models/u2net_portrait/u2net_portrait.pth'

    ONNX_PATH = "/Users/ttjiaa/Pictures/Code/ml/converted/my_model.onnx"

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = [image_dir],
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(512),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------

    print("...load U2NET---173.6 MB")
    net = U2NET(3,1)

    net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for _, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", image_dir.split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        torch.onnx.export(
            model=net,
            args=inputs_test, 
            f=ONNX_PATH, # where should it be saved
            verbose=False,
            export_params=True,
            do_constant_folding=False,  # fold constant values for optimization
            # do_constant_folding=True,   # fold constant values for optimization
            input_names=['input'],
            output_names=['output'],
            opset_version=10
        )
        onnx_model = onnx.load(ONNX_PATH)
        onnx.checker.check_model(onnx_model)

if __name__ == "__main__":
    main()
