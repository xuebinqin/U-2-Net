import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
import argparse
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

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir, colored=False):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split("/")[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

    if colored:
        oriimg = Image.open(image_name)
        bin_image = predict_np*255
        bin_image = Image.fromarray(bin_image).convert('RGB')
        bin_image = bin_image.resize((oriimg.width, oriimg.height), resample=Image.BILINEAR)
        bin_image = np.array(bin_image)
        bin_image = np.where(bin_image > 200, 1, 0)
        colored_img = bin_image * np.array(oriimg)
        colored_img = Image.fromarray(colored_img.astype(np.uint8))
        img = colored_img.convert("RGBA")
        datas = img.getdata()
        newData = []
        for item in datas:
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                newData.append((0, 0, 0, 0))
            else:
                newData.append(item)
        img.putdata(newData)
        img.save(d_dir+imidx+'_COLORED.png', 'PNG')

def main(colored=False):

    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp


    image_dir = './test_data/test_images/'
    prediction_dir = './test_data/' + model_name + '_results/'
    model_dir = './saved_models/'+ model_name + '/' + model_name + '.pth'

    img_name_list = glob.glob(image_dir + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split("/")[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        save_output(img_name_list[i_test],pred,prediction_dir, colored=colored)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--colored', default=False, type=bool, dest='colored', help='Save the colored version of the result. Default=False.')
    args = parser.parse_args()
    main(args.colored)
