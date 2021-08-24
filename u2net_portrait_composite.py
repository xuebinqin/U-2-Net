import os
from skimage import io, transform
from skimage.filters import gaussian
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

import argparse

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir,sigma=2,alpha=0.5):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    image = io.imread(image_name)
    pd = transform.resize(predict_np,image.shape[0:2],order=2)
    pd = pd/(np.amax(pd)+1e-8)*255
    pd = pd[:,:,np.newaxis]

    print(image.shape)
    print(pd.shape)

    ## fuse the orignal portrait image and the portraits into one composite image
    ## 1. use gaussian filter to blur the orginal image
    sigma=sigma
    image = gaussian(image, sigma=sigma, preserve_range=True)

    ## 2. fuse these orignal image and the portrait with certain weight: alpha
    alpha = alpha
    im_comp = image*alpha+pd*(1-alpha)

    print(im_comp.shape)


    img_name = image_name.split(os.sep)[-1]
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    io.imsave(d_dir+'/'+imidx+'_sigma_' + str(sigma) + '_alpha_' + str(alpha) + '_composite.png',im_comp)

def main():

    parser = argparse.ArgumentParser(description="image and portrait composite")
    parser.add_argument('-s',action='store',dest='sigma')
    parser.add_argument('-a',action='store',dest='alpha')
    args = parser.parse_args()
    print(args.sigma)
    print(args.alpha)
    print("--------------------")

    # --------- 1. get image path and name ---------
    model_name='u2net_portrait'#u2netp


    image_dir = './test_data/test_portrait_images/your_portrait_im'
    prediction_dir = './test_data/test_portrait_images/your_portrait_results'
    if(not os.path.exists(prediction_dir)):
        os.mkdir(prediction_dir)

    model_dir = './saved_models/u2net_portrait/u2net_portrait.pth'

    img_name_list = glob.glob(image_dir+'/*')
    print("Number of images: ", len(img_name_list))

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
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

    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = 1.0 - d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        save_output(img_name_list[i_test],pred,prediction_dir,sigma=float(args.sigma),alpha=float(args.alpha))

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
