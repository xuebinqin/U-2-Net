import io
from typing import List, Union, Callable

import cv2
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

from segment_image.session_factory import new_session

onnx_session = new_session("u2net")