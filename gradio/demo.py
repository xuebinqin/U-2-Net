import cv2
import paddlehub as hub
import gradio as gr
import torch

# Images
torch.hub.download_url_to_file('https://cdn.pixabay.com/photo/2018/08/12/16/59/ara-3601194_1280.jpg', 'parrot.jpg')
torch.hub.download_url_to_file('https://cdn.pixabay.com/photo/2016/10/21/14/46/fox-1758183_1280.jpg', 'fox.jpg')

model = hub.Module(name='U2Net')

def infer(img):
  result = model.Segmentation(
      images=[cv2.imread(img.name)],
      paths=None,
      batch_size=1,
      input_size=320,
      output_dir='output',
      visualization=True)
  return result[0]['front'][:,:,::-1], result[0]['mask']

inputs = gr.inputs.Image(type='file', label="Original Image")
outputs = [
           gr.outputs.Image(type="numpy",label="Front"),
           gr.outputs.Image(type="numpy",label="Mask")
           ]

title = "U^2-Net"
description = "demo for U^2-Net. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2005.09007'>U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection</a> | <a href='https://github.com/xuebinqin/U-2-Net'>Github Repo</a></p>"

examples = [
  ['fox.jpg'],
  ['parrot.jpg']
]

gr.Interface(infer, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()