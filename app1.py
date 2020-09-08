import os
import io
import uuid
import shutil
import sys

import threading
import time
from queue import Empty, Queue

import cv2
from flask import Flask, render_template, flash, send_file, request, jsonify, url_for
from PIL import Image
import numpy as np

app = Flask(__name__, template_folder="templates", static_url_path="/static")

# Init Cartoonizer and load its weights
wb_cartoonizer = WB_Cartoonize(os.path.abspath("test_code/saved_models/"))

requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1
##################################################################


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/predict")
def predict():
    print('req')
    return render_template("hi.html")


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=80)