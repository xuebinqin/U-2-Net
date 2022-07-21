import io
from typing import List, Union, Callable

import cv2
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

from segment_image.session_factory import new_session

<<<<<<< HEAD
onnx_session = new_session("u2net")
=======
session = new_session("u2net")


def naive_cutout(img: PILImage, mask: PILImage) -> PILImage:
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask)
    return cutout


def get_concat_v_multi(imgs: List[PILImage]) -> PILImage:
    pivot = imgs.pop(0)
    for im in imgs:
        pivot = get_concat_v(pivot, im)
    return pivot


def get_concat_v(img1: PILImage, img2: PILImage) -> PILImage:
    dst = Image.new("RGBA", (img1.width, img1.height + img2.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (0, img1.height))
    return dst


def post_process_mask(mask:np.ndarray)->np.ndarray:
    '''
    Post Process the mask for a smooth boundary by applying Morphological Operations
    https://www.sciencedirect.com/science/article/pii/S2352914821000757
    args:
        mask: Binary Numpy Mask
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask = cv2.GaussianBlur(mask, (5,5), sigmaX = 1.5, sigmaY = 1.5, borderType = cv2.BORDER_DEFAULT) # Blur
    mask = np.where( mask < 127, 0, 255).astype(np.uint8) # convert again to binary
    return mask


import io
from typing import List, Union

import cv2
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

from .session_factory import new_session

session = new_session("u2net")


def naive_cutout(img: PILImage, mask: PILImage) -> PILImage:
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask)
    return cutout


def get_concat_v_multi(imgs: List[PILImage]) -> PILImage:
    pivot = imgs.pop(0)
    for im in imgs:
        pivot = get_concat_v(pivot, im)
    return pivot


def get_concat_v(img1: PILImage, img2: PILImage) -> PILImage:
    dst = Image.new("RGBA", (img1.width, img1.height + img2.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (0, img1.height))
    return dst


def post_process_mask(mask:np.ndarray)->np.ndarray:
    '''
    Post Process the mask for a smooth boundary by applying Morphological Operations
    https://www.sciencedirect.com/science/article/pii/S2352914821000757
    args:
        mask: Binary Numpy Mask
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask = cv2.GaussianBlur(mask, (5,5), sigmaX = 1.5, sigmaY = 1.5, borderType = cv2.BORDER_DEFAULT) # Blur
    mask = np.where( mask < 127, 0, 255).astype(np.uint8) # convert again to binary
    return mask


def generate_mask(img: Union[bytes, PILImage, np.ndarray],
    post_process:Callable = None,
    return_mask_only:bool = False) -> Union[bytes, PILImage, np.ndarray]:

    masks = session.predict(img)
    cutouts = []

    for mask in masks:
        if post_process is not None:
            mask = Image.fromarray(post_process_mask(np.array(mask))) # Apply post processing to mask
        
        if return_mask_only: return mask
        
        cutout = naive_cutout(img, mask)

        cutouts.append(cutout)

    if len(cutouts) > 0:
        cutout = get_concat_v_multi(cutouts)
    else:
        cutout = img

    return cutout
>>>>>>> c7d5d5ef17c6726f48c16a3c572fb75ae2bee7cf
