import cv2
import paddlehub as hub
import gradio as gr

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

title = "Artline"
description = "demo for OpenAI's CLIP. To use it, simply upload your image, or click one of the examples to load them and optionally add a text label seperated by commas to help clip classify the image better. Read more at the links below."
article = "<p style='text-align: center'><a href='https://openai.com/blog/clip/'>CLIP: Connecting Text and Images</a> | <a href='https://github.com/openai/CLIP'>Github Repo</a></p>"


gr.Interface(infer, inputs, outputs, title=title, description=description, article=article).launch(debug=True)