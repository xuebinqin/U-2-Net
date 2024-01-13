<p align="center">
  <img width="320" height="320" src="figures/U2Net_Logo.png">
  
  <h1 align="center">U<sup>2</sup>-Net: U Square Net</h1>
    
</p>

This is the official repo for our paper **U<sup>2</sup>-Net(U square net)** published in Pattern Recognition 2020:

## [U<sup>2</sup>-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/pdf/2005.09007.pdf)
[Xuebin Qin](https://xuebinqin.github.io/), [Zichen Zhang](https://webdocs.cs.ualberta.ca/~zichen2/), [Chenyang Huang](https://chenyangh.com/), [Masood Dehghan](https://sites.google.com/view/masooddehghan), [Osmar R. Zaiane](http://webdocs.cs.ualberta.ca/~zaiane/) and [Martin Jagersand](https://webdocs.cs.ualberta.ca/~jag/)


__Contact__: xuebin[at]ualberta[dot]ca

## Updates !!!

** (2022-Aug.-24) ** We are glad to announce that our U<sup>2</sup>-Net published in Pattern Recognition has been awarded the 2020 Pattern Recognition BEST PAPER AWARD !!!
![u2net-best-paper](figures/u2net-best-paper.jpg)

** (2022-Aug.-17) **
Our U<sup>2</sup>-Net models are now available on [PlayTorch](https://playtorch.dev/), where you can build your own demo and run it on your Android/iOS phone. Try out this demo on [![PlayTorch Demo](https://github.com/facebookresearch/playtorch/blob/main/website/static/assets/playtorch_badge.svg)](https://playtorch.dev/snack/@playtorch/u2net/) and bring your ideas about U<sup>2</sup>-Net to truth in minutes!

** (2022-Jul.-5)** Our new work **Highly Accurate Dichotomous Image Segmentation (DIS) [**Project Page**](https://xuebinqin.github.io/dis/index.html), [**Github**](https://github.com/xuebinqin/DIS) is accepted by ECCV 2022. Our code and dataset will be released before July 17th, 2022. Please be aware of our updates. 
![ship-demo](figures/ship-demo.gif)
![bg-removal](figures/bg-removal.gif)
![view-move](figures/view-move.gif)
![motor-demo](figures/motor-demo.gif)

** (2022-Jun.-3)** Thank [**Adir Kol**](https://github.com/adirkol) for sharing the iOS App [**3D Photo Creator**](https://apps.apple.com/us/app/3d-photo-creator/id1619676262) based on our U<sup>2</sup>-Net.
![portrait-ios-app](figures/3d-photo-re.jpg)

** (2022-Mar.-31)** Thank [**Hikaru Tsuyumine**] for implementing the iOS App [**Portrait Drawing**](https://apps.apple.com/us/app/portrait-drawing/id1623269600) based on our U<sup>2</sup>-Net portrait generation model.
![portrait-ios-app](figures/portrait-ios-app.jpg)

** (2022-Apr.-12)** Thank [**Kevin Shah**](https://github.com/ioskevinshah) for providing us a great iOS App [**Lensto**](https://apps.apple.com/in/app/lensto-background-changer/id1574844033), ([**Demo Video**](https://www.youtube.com/shorts/jWwUiKZjfok)), based on U<sup>2</sup>-Net.
![lensto](figures/lensto.png)

** (2022-Mar.-31)** Our U<sup>2</sup>-Net model is also integrated by [**Hotpot.ai**](https://hotpot.ai/) for art design.
![hotpot](figures/hotpot.png)

** (2022-Mar-19)** Thank [**Kikedao**](https://github.com/Kikedao) for providing a fantastic webapp [**Silueta**](https://silueta.me/) based on U<sup>2</sup>-Net. More details can be found at [**https://github.com/xuebinqin/U-2-Net/issues/295**](https://github.com/xuebinqin/U-2-Net/issues/295).
![silueta](figures/silueta.png) 

** (2022-Mar-17)** Thank [**Ezaldeen Sahb**](https://github.com/Ezaldeen99/BackgroundRemoval) for implementing the iOS library for image background removal based on U<sup>2</sup>-Net, which will greatly facilitate the developing of mobile apps.
![close-seg](figures/swift-u2net.jpeg) 

<!-- ** (2022-Mar-10)** Thank [**Doron Adler**](https://github.com/Norod/U-2-Net-StyleTransfer) for training the awesome style transfer U<sup>2</sup>-Net.
![style-trans](figures/style-trans.JPG)  -->

** (2022-Mar-8)** Thank [**Levin Dabhi**](https://github.com/levindabhi/cloth-segmentation) for training the amazing clothes segmentation U<sup>2</sup>-Net.
![close-seg](figures/close-seg.jpg) 

** (2022-Mar-3)** Thank [**Renato Violin**](https://github.com/renatoviolin/bg-remove-augment) for providing an awesome webapp for image background removal and replacement based on our U<sup>2</sup>-Net.
![bg-rm-aug](figures/bg-rm-aug.gif) 

**(2021-Dec-21)** This [**blog**](https://rockyshikoku.medium.com/u2net-to-coreml-machine-learning-segmentation-on-iphone-eac0c721d67b) clearly describes the way of converting U<sup>2</sup>-Net to [**CoreML**](https://github.com/john-rocky/CoreML-Models) and running it on iphone. 

**(2021-Nov-28)** Interesting Sky Segmentation models developed by [**xiongzhu**](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing) using U<sup>2</sup>-Net. 

![im_sky_segmentation](figures/sky-seg.png)

**(2021-Nov-28)** Awesome image editing app [**Pixelmator pro**](https://www.pixelmator.com/pro/) uses U<sup>2</sup>-Net as one of its background removal models. 

![im_sky_segmentation](figures/pixelmator.jpg)

**(2021-Aug-24)** We played a bit more about fusing the orignal image and the generated portraits to composite different styles. You can <br/> 
(1) Download this repo by
```
git clone https://github.com/NathanUA/U-2-Net.git
```
(2) Download the u2net_portrait.pth from [**GoogleDrive**](https://drive.google.com/file/d/1IG3HdpcRiDoWNookbncQjeaPN28t90yW/view?usp=sharing) or [**Baidu Pan(提取码：chgd)**](https://pan.baidu.com/s/1BYT5Ts6BxwpB8_l2sAyCkw)model and put it into the directory: ```./saved_models/u2net_portrait/```, <br/>
(3) run the code by command 
```
python u2net_portrait_composite.py -s 20 -a 0.5
```
,where ``-s`` indicates the sigma of gaussian function for blurring the orignal image and ``-a`` denotes the alpha weights of the orignal image when fusing them. <br/>

![im_portrait_composite](figures/im_composite.jpg)

**(2021-July-16)** A new [background removal webapp](https://remove-background.net/) developed by Изатоп Василий. 

![rm_bg](figures/rm_bg.JPG)

**(2021-May-26)** Thank [**Dang Quoc Quy**](https://github.com/quyvsquy) for his [**Art Transfer APP**](https://play.google.com/store/apps/details?id=com.quyvsquy.arttransfer) built upon U<sup>2</sup>-Net.

<!---![art_transfer](figures/art_transfer.JPG)--->

**(2021-May-5)** Thank [**AK391**](https://github.com/AK391) for sharing his [**Gradio Web Demo of U<sup>2</sup>-Net**](https://gradio.app/hub/AK391/U-2-Net).

![gradio_web_demo](figures/gradio_web_demo.jpg)


**(2021-Apr-29)** Thanks [**Jonathan Benavides Vallejo**](https://www.linkedin.com/in/jonathanbv/) for releasing his App [**LensOCR: Extract Text & Image**](https://apps.apple.com/ch/app/lensocr-extract-text-image/id1549961729?l=en&mt=12), which uses U<sup>2</sup>-Net for extracting the image foreground.

![LensOCR APP](figures/LensOCR.jpg)

**(2021-Apr-18)** Thanks [**Andrea Scuderi**](https://www.linkedin.com/in/andreascuderi/) for releasing his App [**Clipping Camera**](https://apps.apple.com/us/app/clipping-camera/id1548192169?ign-mpt=uo%3D2), which is an U<sup>2</sup>-Net driven realtime camera app and "is able to detect relevant object from the scene and clip them to apply fancy filters". 

![Clipping Camera APP](figures/clipping_camera.jpg)

**(2021-Mar-17)** [**Dennis Bappert**](https://github.com/dennisbappert) re-trained the U<sup>2</sup>-Net model for [**human portrait matting**](https://github.com/dennisbappert/u-2-net-portrait). The results look very promising and he also provided the details of the training process and data generation(and augmentation) strategy, which are inspiring.

**(2021-Mar-11)** Dr. Tim developed a [**video version rembg**](https://github.com/ecsplendid/rembg-greenscreen) for removing video backgrounds using U<sup>2</sup>-Net. The awesome demo results can be found on [**YouTube**](https://www.youtube.com/watch?v=4NjqR2vCV_k).

**(2021-Mar-02)** We found some other interesting applications of our U<sup>2</sup>-Net including [**MOJO CUT**](https://play.google.com/store/apps/details?id=com.innoria.magicut&hl=en_CA&gl=US), [**Real-Time Background Removal on Iphone**](https://www.linkedin.com/feed/update/urn:li:activity:6752303661705170944/?updateEntityUrn=urn%3Ali%3Afs_feedUpdate%3A%28V2%2Curn%3Ali%3Aactivity%3A6752303661705170944%29), [**Video Background Removal**](https://nisargkapkar.hashnode.dev/image-and-video-background-removal-using-deep-learning), [**Another Online Portrait Generation Demo on AWS**](http://s3-website-hosting-u2net.s3-website-eu-west-1.amazonaws.com/), [**AI Scissor**](https://qooba.net/2020/09/11/ai-scissors-sharp-cut-with-neural-networks/).

**(2021-Feb-15)** We just released an online demo [**http://profu.ai**](http://profu.ai) for the portrait generation. Please feel free to give it a try and provide any suggestions or comments. <br/>
![Profuai](figures/profuai.png) <br/>

**(2021-Feb-06)** Recently, some people asked the problem of using U<sup>2</sup>-Net for human segmentation, so we trained another example model for human segemntation based on [**Supervisely Person Dataset**](https://supervise.ly/explore/projects/supervisely-person-dataset-23304/datasets). <br/>

(1) To run the human segmentation model, please first downlowd the [**u2net_human_seg.pth**](https://drive.google.com/file/d/1m_Kgs91b21gayc2XLW0ou8yugAIadWVP/view?usp=sharing) model weights into ``` ./saved_models/u2net_human_seg/```. <br/>
(2) Prepare the to-be-segmented images into the corresponding directory, e.g. ```./test_data/test_human_images/```. <br/>
(3) Run the inference by command: ```python u2net_human_seg_test.py``` and the results will be output into the corresponding dirctory, e.g. ```./test_data/u2net_test_human_images_results/```<br/>
[**Notes: Due to the labeling accuracy of the Supervisely Person Dataset, the human segmentation model (u2net_human_seg.pth) here won't give you hair-level accuracy. But it should be more robust than u2net trained with DUTS-TR dataset on general human segmentation task. It can be used for human portrait segmentation, human body segmentation, etc.**](https://github.com/NathanUA/U-2-Net)<br/>

![Human Image Segmentation](figures/human_seg.png) <br/>
![Human Video](figures/human_seg_video.gif)
![Human Video Results](figures/human_seg_results.gif)

**(2020-Dec-28)** Some interesting applications and useful tools based on U<sup>2</sup>-Net: <br/>
(1) [**Xiaolong Liu**](https://github.com/LiuXiaolong19920720) developed several very interesting applications based on U<sup>2</sup>-Net including [**Human Portrait Drawing**](https://www.cvpy.net/studio/cv/func/DeepLearning/sketch/sketch/page/)(As far as I know, Xiaolong is the first one who uses U<sup>2</sup>-Net for portrait generation), [**image matting**](https://www.cvpy.net/studio/cv/func/DeepLearning/matting/matting/page/) and [**so on**](https://www.cvpy.net/). <br/>
(2) [**Vladimir Seregin**](https://github.com/peko/nn-lineart) developed an interesting tool, [**NN based lineart**](https://peko.github.io/nn-lineart/), for comparing the portrait results of U<sup>2</sup>-Net and that of another popular model, [**ArtLine**](https://github.com/vijishmadhavan/ArtLine), developed by [**Vijish Madhavan**](https://github.com/vijishmadhavan). <br/>
(3) [**Daniel Gatis**](https://github.com/danielgatis/rembg) built a python tool, [**Rembg**](https://pypi.org/project/rembg/), for image backgrounds removal based on U<sup>2</sup>-Net. I think this tool will greatly facilitate the application of U<sup>2</sup>-Net in different fields. <br/>
![REMBG](figures/rembg.png)

**(2020-Nov-21)** Recently, we found an interesting application of U<sup>2</sup>-Net for [**human portrait drawing**](https://www.pythonf.cn/read/141098). Therefore, we trained another model for this task based on the [**APDrawingGAN dataset**](https://github.com/yiranran/APDrawingGAN).

![Sample Results: Kids](figures/portrait_kids.png)

![Sample Results: Ladies](figures/portrait_ladies.png)

![Sample Results: Men](figures/portrait_men.png)

### Usage for portrait generation
1. Clone this repo to local
```
git clone https://github.com/NathanUA/U-2-Net.git
```

2. Download the u2net_portrait.pth from [**GoogleDrive**](https://drive.google.com/file/d/1IG3HdpcRiDoWNookbncQjeaPN28t90yW/view?usp=sharing) or [**Baidu Pan(提取码：chgd)**](https://pan.baidu.com/s/1BYT5Ts6BxwpB8_l2sAyCkw)model and put it into the directory: ```./saved_models/u2net_portrait/```.

3. Run on the testing set. <br/>
(1) Download the train and test set from [**APDrawingGAN**](https://github.com/yiranran/APDrawingGAN). These images and their ground truth are stitched side-by-side (512x1024). You need to split each of these images into two 512x512 images and put them into ```./test_data/test_portrait_images/portrait_im/```. You can also download the split testing set on [GoogleDrive](https://drive.google.com/file/d/1NkTsDDN8VO-JVik6VxXyV-3l2eo29KCk/view?usp=sharing). <br/>
(2) Running the inference with command ```python u2net_portrait_test.py``` will ouptut the results into ```./test_data/test_portrait_images/portrait_results```. <br/>

4. Run on your own dataset. <br/>
(1) Prepare your images and put them into ```./test_data/test_portrait_images/your_portrait_im/```. [**To obtain enough details of the protrait, human head region in the input image should be close to or larger than 512x512. The head background should be relatively clear.**](https://github.com/NathanUA/U-2-Net) <br/>
(2) Run the prediction by command ```python u2net_portrait_demo.py``` will outputs the results to ```./test_data/test_portrait_images/your_portrait_results/```. <br/>
(3) The difference between ```python u2net_portrait_demo.py``` and ```python u2net_portrait_test.py``` is that we added a simple [**face detection**](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html) step before the portrait generation in ```u2net_portrait_demo.py```.  Because the testing set of APDrawingGAN are normalized and cropped to 512x512 for including only heads of humans, while our own dataset may varies with different resolutions and contents. Therefore, the code ```python u2net_portrait_demo.py``` will detect the biggest face from the given image and then crop, pad and resize the ROI to 512x512 for feeding to the network. The following figure shows how to take your own photos for generating high quality portraits.

**(2020-Sep-13)** Our U<sup>2</sup>-Net based model is the **6th** in [**MICCAI 2020 Thyroid Nodule Segmentation Challenge**](https://tn-scui2020.grand-challenge.org/Resultannouncement/).

**(2020-May-18)** The official paper of our **U<sup>2</sup>-Net (U square net)** ([**PDF in elsevier**(free until July 5 2020)](https://www.sciencedirect.com/science/article/pii/S0031320320302077?dgcid=author), [**PDF in arxiv**](http://arxiv.org/abs/2005.09007)) is now available. If you are not able to access that, please feel free to drop me an email.

**(2020-May-16)** We fixed the upsampling issue of the network. Now, the model should be able to handle **arbitrary input size**. (Tips: This modification is to facilitate the retraining of U<sup>2</sup>-Net on your own datasets. When using our pre-trained model on SOD datasets, please keep the input size as 320x320 to guarantee the performance.)

**(2020-May-16)** We highly appreciate **Cyril Diagne** for building this fantastic AR project: [**AR Copy and Paste**](https://github.com/cyrildiagne/ar-cutpaste) using our **U<sup>2</sup>-Net** (Qin *et al*, PR 2020) and [**BASNet**](https://github.com/NathanUA/BASNet)(Qin *et al*, CVPR 2019). The [**demo video**](https://twitter.com/cyrildiagne/status/1256916982764646402) in twitter has achieved over **5M** views, which is phenomenal and shows us more application possibilities of SOD.

## U<sup>2</sup>-Net Results (176.3 MB)

![U<sup>2</sup>-Net Results](figures/u2netqual.png)


## Our previous work: [BASNet (CVPR 2019)](https://github.com/NathanUA/BASNet)

## Required libraries

Python 3.6  
numpy 1.15.2  
scikit-image 0.14.0  
python-opencv
PIL 5.2.0  
PyTorch 0.4.0  
torchvision 0.2.1  
glob  

## Usage for salient object detection
1. Clone this repo
```
git clone https://github.com/NathanUA/U-2-Net.git
```
2. Download the pre-trained model u2net.pth (176.3 MB) from [**GoogleDrive**](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?usp=sharing) or [**Baidu Pan 提取码: pf9k**](https://pan.baidu.com/s/1WjwyEwDiaUjBbx_QxcXBwQ) or u2netp.pth (4.7 MB) from [**GoogleDrive**](https://drive.google.com/file/d/1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy/view?usp=sharing) or [**Baidu Pan 提取码: 8xsi**](https://pan.baidu.com/s/10tW12OlecRpE696z8FxdNQ) and put it into the dirctory './saved_models/u2net/' and './saved_models/u2netp/'

3.  Cd to the directory 'U-2-Net', run the train or inference process by command: ```python u2net_train.py```
or ```python u2net_test.py``` respectively. The 'model_name' in both files can be changed to 'u2net' or 'u2netp' for using different models.  

 We also provide the predicted saliency maps ([u2net results](https://drive.google.com/file/d/1mZFWlS4WygWh1eVI8vK2Ad9LrPq4Hp5v/view?usp=sharing),[u2netp results](https://drive.google.com/file/d/1j2pU7vyhOO30C2S_FJuRdmAmMt3-xmjD/view?usp=sharing)) for datasets SOD, ECSSD, DUT-OMRON, PASCAL-S, HKU-IS and DUTS-TE.


## U<sup>2</sup>-Net Architecture

![U<sup>2</sup>-Net architecture](figures/U2NETPR.png)


## Quantitative Comparison

![Quantitative Comparison](figures/quan_1.png)

![Quantitative Comparison](figures/quan_2.png)


## Qualitative Comparison

![Qualitative Comparison](figures/qual.png?raw=true)

## Citation
```
@InProceedings{Qin_2020_PR,
title = {U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection},
author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar and Jagersand, Martin},
journal = {Pattern Recognition},
volume = {106},
pages = {107404},
year = {2020}
}
```
