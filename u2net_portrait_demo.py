import cv2
import torch
from model import U2NET
from torch.autograd import Variable
import numpy as np
from glob import glob
import os

def detect_single_face(face_cascade,img):
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if(len(faces)==0):
        print("Warming: no face detection, the portrait u2net will run on the whole image!")
        return None

    # filter to keep the largest face
    wh = 0
    idx = 0
    for i in range(0,len(faces)):
        (x,y,w,h) = faces[i]
        if(wh<w*h):
            idx = i
            wh = w*h

    return faces[idx]

# crop, pad and resize face region to 512x512 resolution
def crop_face(img, face):

    # no face detected, return the whole image and the inference will run on the whole image
    if(face is None):
        return img
    (x, y, w, h) = face

    height,width = img.shape[0:2]

    # crop the face with a bigger bbox
    hmw = h - w
    # hpad = int(h/2)+1
    # wpad = int(w/2)+1

    l,r,t,b = 0,0,0,0
    lpad = int(float(w)*0.4)
    left = x-lpad
    if(left<0):
        l = lpad-x
        left = 0

    rpad = int(float(w)*0.4)
    right = x+w+rpad
    if(right>width):
        r = right-width
        right = width

    tpad = int(float(h)*0.6)
    top = y - tpad
    if(top<0):
        t = tpad-y
        top = 0

    bpad  = int(float(h)*0.2)
    bottom = y+h+bpad
    if(bottom>height):
        b = bottom-height
        bottom = height


    im_face = img[top:bottom,left:right]
    if(len(im_face.shape)==2):
        im_face = np.repeat(im_face[:,:,np.newaxis],(1,1,3))

    im_face = np.pad(im_face,((t,b),(l,r),(0,0)),mode='constant',constant_values=((255,255),(255,255),(255,255)))

    # pad to achieve image with square shape for avoding face deformation after resizing
    hf,wf = im_face.shape[0:2]
    if(hf-2>wf):
        wfp = int((hf-wf)/2)
        im_face = np.pad(im_face,((0,0),(wfp,wfp),(0,0)),mode='constant',constant_values=((255,255),(255,255),(255,255)))
    elif(wf-2>hf):
        hfp = int((wf-hf)/2)
        im_face = np.pad(im_face,((hfp,hfp),(0,0),(0,0)),mode='constant',constant_values=((255,255),(255,255),(255,255)))

    # resize to have 512x512 resolution
    im_face = cv2.resize(im_face, (512,512), interpolation = cv2.INTER_AREA)

    return im_face

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def inference(net,input):

    # normalize the input
    tmpImg = np.zeros((input.shape[0],input.shape[1],3))
    input = input/np.max(input)

    tmpImg[:,:,0] = (input[:,:,2]-0.406)/0.225
    tmpImg[:,:,1] = (input[:,:,1]-0.456)/0.224
    tmpImg[:,:,2] = (input[:,:,0]-0.485)/0.229

    # convert BGR to RGB
    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = tmpImg[np.newaxis,:,:,:]
    tmpImg = torch.from_numpy(tmpImg)

    # convert numpy array to torch tensor
    tmpImg = tmpImg.type(torch.FloatTensor)

    if torch.cuda.is_available():
        tmpImg = Variable(tmpImg.cuda())
    else:
        tmpImg = Variable(tmpImg)

    # inference
    d1,d2,d3,d4,d5,d6,d7= net(tmpImg)

    # normalization
    pred = 1.0 - d1[:,0,:,:]
    pred = normPRED(pred)

    # convert torch tensor to numpy array
    pred = pred.squeeze()
    pred = pred.cpu().data.numpy()

    del d1,d2,d3,d4,d5,d6,d7

    return pred

def main():

    # get the image path list for inference
    im_list = glob('./test_data/test_portrait_images/your_portrait_im/*')
    print("Number of images: ",len(im_list))
    # indicate the output directory
    out_dir = './test_data/test_portrait_images/your_portrait_results'
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)

    # Load the cascade face detection model
    face_cascade = cv2.CascadeClassifier('./saved_models/face_detection_cv2/haarcascade_frontalface_default.xml')
    # u2net_portrait path
    model_dir = './saved_models/u2net_portrait/u2net_portrait.pth'

    # load u2net_portrait model
    net = U2NET(3,1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # do the inference one-by-one
    for i in range(0,len(im_list)):
        print("--------------------------")
        print("inferencing ", i, "/", len(im_list), im_list[i])

        # load each image
        img = cv2.imread(im_list[i])
        height,width = img.shape[0:2]
        face = detect_single_face(face_cascade,img)
        im_face = crop_face(img, face)
        im_portrait = inference(net,im_face)

        # save the output
        cv2.imwrite(out_dir+"/"+im_list[i].split('/')[-1][0:-4]+'.png',(im_portrait*255).astype(np.uint8))

if __name__ == '__main__':
    main()
