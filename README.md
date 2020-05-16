# U^2-Net

The code for our newly accepted paper in Pattern Recognition 2020:
## [U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://www.sciencedirect.com/science/article/pii/S0031320320302077?dgcid=author), [Xuebin Qin](https://webdocs.cs.ualberta.ca/~xuebin/), [Zichen Zhang](https://webdocs.cs.ualberta.ca/~zichen2/), [Chenyang Huang](https://chenyangh.com/), [Masood Dehghan](https://sites.google.com/view/masooddehghan), [Osmar R. Zaiane](http://webdocs.cs.ualberta.ca/~zaiane/) and [Martin Jagersand](https://webdocs.cs.ualberta.ca/~jag/).

__Contact__: xuebin[at]ualberta[dot]ca

## News !!!

**(2020-May-16)** The official paper of **U^2-Net (U square net)** [PDF in elsevier](https://www.sciencedirect.com/science/article/pii/S0031320320302077?dgcid=author) is now available. If you are not able to access that, please feel free to drop me an email.

**(2020-May-16)** We fixed the upsampling issue of the network. Now, the model should be able to handle **arbitrary input size**. (Tips: This modification is to facilitate the retraining of U^2-Net on your own datasets. When using our pre-trained model on SOD datasets, please keep the input size as 320x32 to guarantee the performance.)

**(2020-May-16)** We highly appreciate Cyril Diagne for building this fantastic AR project: [AR Copy and Paste](https://github.com/cyrildiagne/ar-cutpaste) using our **U^2-Net** and [**BASNet**](https://github.com/NathanUA/BASNet). The [demo video](https://twitter.com/cyrildiagne/status/1256916982764646402) in twitter has achieved over **5M** views, which is phenomenal and shows us more probabilities of SOD.

## U^2-Net Results (173.6 MB)

![U^2-Net Results](figures/u2netqual.png)


## Our previous work: [BASNet (CVPR 2019)](https://github.com/NathanUA/BASNet)

## Required libraries

Python 3.6  
numpy 1.15.2  
scikit-image 0.14.0  
PIL 5.2.0  
PyTorch 0.4.0  
torchvision 0.2.1  
glob  

## Usage
1. Clone this repo
```
git clone https://github.com/NathanUA/U-2-Net.git
```
2. Download the pre-trained model [u2net.pth (173.6 MB)](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?usp=sharing) or [u2netp.pth (4.7 MB)](https://drive.google.com/file/d/1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy/view?usp=sharing) and put it into the dirctory './saved_models/u2net/' and './saved_models/u2netp/'

3.  Cd to the directory 'U-2-Net', run the train or inference process by command: ```python u2net_train.py```
or ```python u2net_test.py``` respectively. The 'model_name' in both files can be changed to 'u2net' or 'u2netp' for using different models.  

 We also provide the predicted saliency maps ([u2net results](https://drive.google.com/file/d/1mZFWlS4WygWh1eVI8vK2Ad9LrPq4Hp5v/view?usp=sharing),[u2netp results](https://drive.google.com/file/d/1j2pU7vyhOO30C2S_FJuRdmAmMt3-xmjD/view?usp=sharing)) for datasets SOD, ECSSD, DUT-OMRON, PASCAL-S, HKU-IS and DUTS-TE.


## U^2-Net Architecture

![U^2-Net architecture](figures/U2NETPR.png)


## Quantitative Comparison

![Quantitative Comparison](figures/quan_1.png)

![Quantitative Comparison](figures/quan_2.png)


## Qualitative Comparison

![Qualitative Comparison](figures/qual.png?raw=true)


## Citation
```
@InProceedings{Qin_2020_PR,
author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar and Jagersand, Martin},
title = {U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection},
booktitle = {Pattern Recognition},
year = {2020}
}
```
