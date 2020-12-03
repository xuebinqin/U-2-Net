import os
import tensorflow as tf

TF_PATH = "/Users/ttjiaa/Pictures/Code/ml/converted/my_tf_model.pb"
TFLITE_PATH = "/Users/ttjiaa/Pictures/Code/ml/converted/my_tf_model.tflite"

# protopuf needs your virtual environment to be explictly exported in the path
os.environ["PATH"] = "/opt/miniconda3/envs/convert/bin:/opt/miniconda3/bin:/usr/local/sbin:...."

converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)

tf_lite_model = converter.convert()

# Save the model.
with open(TFLITE_PATH, 'wb') as f:
  f.write(tf_lite_model)
