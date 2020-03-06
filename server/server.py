from flask import Flask, render_template, request, redirect, abort
from werkzeug.utils import secure_filename

from redis import Redis
import rq

import os
import torch
import torchvision.models as models

app = Flask(__name__)


# CONSTANTS
RQ_WORKER_NAME = "transfer-task"
REDIS_URL = "redis://"
ALLOWED_TYPES = ["image/jpeg"]


# TASKS:
# 1. Write function that loads the model and prepares it on server launch
# 1.1 Make it so it only loads on >>server launch<< not on every user connection if you know how
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.vgg19(pretrained=True).features.to(device).eval()


# 2. Write simple homepage that let's you upload an content and style images
app.secret_key = os.urandom(16)
queue = rq.Queue(RQ_WORKER_NAME, connection=Redis.from_url('redis://'))


# def check_file(filename):
#     ext = os.path.splitext(filename)[-1].lower()
#
#     if ext in ALLOWED_FILE_FORMATS:
#         return True
#     else:
#         return False
#
#
# def save_file(file):
#     try:
#         target = os.path.join(PROJECT_ROOT, UPLOAD_FOLDER)
#
#         if not os.path.isdir(target):
#             os.mkdir(target)
#
#         filename = secure_filename(file.filename)
#         destination = "/".join([target, filename])
#         file.save(destination)
#         return True
#
#     except:
#         return False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run_transfer', methods=["POST"])
def run_transfer():
    style_image = request.files.get('style_image')
    content_image = request.files.get('content_image')

    if style_image.mimetype not in ALLOWED_TYPES \
            or content_image.mimetype not in ALLOWED_TYPES:
        abort(415)
        # TODO


# 3. Write a function that uses style_transfer() function from style_transfer.py to generate new images
# from uploaded content/style images
# 3.1 Write a simple front that shows progress or gives the user information that the image is being generated
# 4. Present the result for user in simple front, maybe add download button so user can download the new image

# def style_transfer_job(style_image, content_image):


if __name__ == "__main__":
        
    app.run(debug=True)
