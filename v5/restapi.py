# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run a Flask REST API exposing a YOLOv5s model
"""

import argparse
import io
from PIL import Image as PILImage
import torch
from flask import Flask, request
from PIL import Image
from detect import run

app = Flask(__name__)

DETECTION_URL = "/v1/object-detection/yolov5n"


@app.route("/kivi", methods=["POST"])
def predict():
    
    # Method 1
    # with request.files["image"] as f:
    #     im = Image.open(io.BytesIO(f.read()))
    data = request.data
    # im =  Image.open(data)
    # print(f"Request data_type: {type(data)}")
    by = bytes(data)
    # img1 = Image.open(io.BytesIO(by))
    # Method 2
    # im_file = request.files["image"]
    # im_bytes = im_file.read()
    im = Image.open(io.BytesIO(by))

    results = model(im, size=640)  # reduce size=320 for faster inference
    # run(binary=by)
    return results.pandas().xyxy[0].to_json(orient="records")

@app.route("/", methods=["POST"])
def det():
    
    # Method 1
    # with request.files["image"] as f:
    #     im = Image.open(io.BytesIO(f.read()))
    data = request.data
    # im =  Image.open(data)
    # print(f"Request data_type: {type(data)}")
    by = bytes(data)
    # img1 = Image.open(io.BytesIO(by))
    # Method 2
    # im_file = request.files["image"]
    # im_bytes = im_file.read()
    im = Image.open(io.BytesIO(by))

    results = model(im, size=640)  # reduce size=320 for faster inference
    # run(binary=by)
    return results.pandas().xyxy[0].to_json(orient="records")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    opt = parser.parse_args()

    # Fix known issue urllib.error.HTTPError 403: rate limit exceeded https://github.com/ultralytics/yolov5/pull/7210
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

    model = torch.hub.load("ultralytics/yolov5", "yolov5n", force_reload=True)  # force_reload to recache
    app.run(host="0.0.0.0", port=opt.port)  # debug=True causes Restarting with stat
