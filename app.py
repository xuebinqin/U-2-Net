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

sys.path.insert(0, "./test_code/")

from cartoonize import WB_Cartoonize

###################################################################
app = Flask(__name__, template_folder="templates", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 35 * 1024 * 1024

DATA_FOLDER = "data"

## Init Cartoonizer and load its weights
wb_cartoonizer = WB_Cartoonize(os.path.abspath("test_code/saved_models/"))

requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1
##################################################################
def convert_bytes_to_image(img_bytes):
    """Convert bytes to numpy array
    Args:
        img_bytes (bytes): Image bytes read from flask.
    Returns:
        [numpy array]: Image numpy array
    """

    pil_image = Image.open(io.BytesIO(img_bytes))
    if pil_image.mode == "RGBA":
        image = Image.new("RGB", pil_image.size, (255, 255, 255))
        image.paste(pil_image, mask=pil_image.split()[3])
    else:
        image = pil_image.convert("RGB")

    image = np.array(image)

    return image


def run(input_file, file_type, f_path):
    try:
        if file_type == "image":
            f_name = str(uuid.uuid4())

            img = input_file.read()

            ## Read Image and convert to PIL (RGB) if RGBA convert appropriately
            image = convert_bytes_to_image(img)

            cartoon_image = wb_cartoonizer.infer(image)

            cartoonized_img_name = os.path.join(f_path, f_name + ".jpg")
            cv2.imwrite(
                cartoonized_img_name, cv2.cvtColor(cartoon_image, cv2.COLOR_RGB2BGR)
            )

            result_path = cartoonized_img_name

            return result_path

        if file_type == "video":
            f_name = input_file.filename

            video = input_file

            original_video_path = os.path.join(f_path, f_name)
            video.save(original_video_path)

            # Slice, Resize and Convert Video to 15fps
            modified_video_path = os.path.join(
                f_path, f_name.split(".")[0] + "_modified.mp4"
            )
            width_resize = 480
            os.system(
                "ffmpeg -hide_banner -loglevel warning -ss 0 -i '{}' -t 10 -filter:v scale={}:-2 -r 15 -c:a copy '{}'".format(
                    os.path.abspath(original_video_path),
                    width_resize,
                    os.path.abspath(modified_video_path),
                )
            )

            # if local then "output_uri" is a file path
            output_uri = wb_cartoonizer.process_video(modified_video_path)

            result_path = output_uri

            return result_path

    except Exception as e:
        print(e)
        return 500


def handle_requests_by_batch():
    try:
        while True:
            requests_batch = []

            while not (
                len(requests_batch)
                >= BATCH_SIZE  # or
                # (len(requests_batch) > 0 #and time.time() - requests_batch[0]['time'] > BATCH_TIMEOUT)
            ):
                try:
                    requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
                except Empty:
                    continue

            batch_outputs = []

            for request in requests_batch:
                batch_outputs.append(
                    run(request["input"][0], request["input"][1], request["input"][2])
                )

            for request, output in zip(requests_batch, batch_outputs):
                request["output"] = output

    except Exception as e:
        while not requests_queue.empty():
            requests_queue.get()
        print(e)


threading.Thread(target=handle_requests_by_batch).start()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        print(requests_queue.qsize())

        if requests_queue.qsize() >= 1:
            return jsonify({"message": "Too Many Requests"}), 429

        input_file = request.files["source"]
        file_type = request.form["file_type"]

        if file_type == "image":
            if input_file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
                return jsonify({"message": "Only support jpeg, jpg or png"}), 400

        else:
            if input_file.content_type not in ["video/mp4"]:
                return jsonify({"message": "Only support mp4"}), 400

        f_id = str(uuid.uuid4())
        f_path = os.path.join(DATA_FOLDER, f_id)
        os.makedirs(f_path, exist_ok=True)

        req = {"input": [input_file, file_type, f_path]}

        requests_queue.put(req)

        while "output" not in req:
            time.sleep(CHECK_INTERVAL)

        if req["output"] == 500:
            return jsonify({"error": "Error! Please upload another file"}), 500

        result_path = req["output"]

        result = send_file(result_path)

        shutil.rmtree(f_path)

        return result

    except Exception as e:
        print(e)

        return jsonify({"message": "Error! Please upload another file"}), 400


@app.route("/health")
def health():
    return "ok"


@app.route("/")
def main():
    return render_template("index.html")


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=80)
