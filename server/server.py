from flask import Flask, url_for, render_template, request, redirect, flash
from pathlib import Path
from werkzeug.utils import secure_filename

import os
import torch
import torchvision.models as models

app = Flask(__name__)


SERVER_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(SERVER_ROOT).parent
IMAGE_FOLDER = 'images'
UPLOAD_FOLDER = 'images/uploads'

# TASKS:
# 1. Write function that loads the model and prepares it on server launch
# 1.1 Make it so it only loads on >>server launch<< not on every user connection if you know how

MEAN_STANDARDIZED = [0.485, 0.456, 0.406]
STD_STANDARDIZED = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
VGG19_MODEL = models.vgg19(pretrained=True).features.to(DEVICE).eval()
IMSIZE = (512, 512) if torch.cuda.is_available() else (300, 300)
ALLOWED_FILE_FORMATS = ['.png', '.jpg', '.jpeg']

# 2. Write simple homepage that let's you upload an content and style images


def check_file(filename):
    ext = os.path.splitext(filename)[-1].lower()

    if ext in ALLOWED_FILE_FORMATS:
        return True
    else:
        return False


def save_file(file):
    try:
        target = os.path.join(PROJECT_ROOT, UPLOAD_FOLDER)

        if not os.path.isdir(target):
            os.mkdir(target)

        filename = secure_filename(file.filename)
        destination = "/".join([target, filename])
        file.save(destination)
        return True

    except:
        return False


@app.route('/')
def index():
    photos_dict = dict()
    img_path = os.path.join(PROJECT_ROOT, UPLOAD_FOLDER)

    if os.path.isdir(img_path):
        for filename in os.listdir(img_path):
            photos_dict[filename] = filename

    print(photos_dict)
    return render_template('index.html', photos=photos_dict)


@app.route('/upload', methods=["POST"])
def upload():
    # Alternatively we could use flash method in order to show the message to the user with error message
    input_file = request.files['inputFile']

    if input_file.filename == "":
        return redirect(url_for('fail_page', reason="No file was selected!"))

    if check_file(input_file.filename):

        if save_file(input_file):
            return render_template("uploaded.html")
        else:
            return redirect(url_for('fail_page', reason="Couldn't save the file"))
    else:
        return redirect(url_for('fail_page', reason='Invalid file format'))


@app.route('/styleTransfer')
def run_style_transfer():
    return render_template("run_style_transfer.html")


@app.route('/fail/<reason>')
def fail_page(reason):
    return render_template("upload_fail.html", reason=reason)

# 3. Write a function that uses style_transfer() function from style_transfer.py to generate new images
# from uploaded content/style images
# 3.1 Write a simple front that shows progress or gives the user information that the image is being generated
# 4. Present the result for user in simple front, maybe add download button so user can download the new image


if __name__ == "__main__":
        
    app.run(debug=True)
