from flask import Flask, url_for, render_template, request
from pathlib import Path


import os
import torch
import torchvision.models as models

app = Flask(__name__)

SERVER_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(SERVER_ROOT).parent

# TASKS:
# 1. Write function that loads the model and prepares it on server launch
# 1.1 Make it so it only loads on >>server launch<< not on every user connection if you know how

MEAN_STANDARDIZED = [0.485, 0.456, 0.406]
STD_STANDARDIZED = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
VGG19_MODEL = models.vgg19(pretrained=True).features.to(DEVICE).eval()
IMSIZE = (512,512) if torch.cuda.is_available() else (300,300)

# 2. Write simple homepage that let's you upload an content and style images

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=["POST"])
def upload():
    ifile = request.files['inputFile']
    target = os.path.join(PROJECT_ROOT, 'images/')
    if not os.path.isdir(target):
        os.mkdir(target)
        
    print(ifile)
    filename = ifile.filename
    destination = "/".join([target, filename])
    print(destination)
    ifile.save(destination)
        
    return render_template("uploaded.html")

# 3. Write a function that uses style_transfer() function from style_transfer.py to generate new images from uploaded content/style images
# 3.1 Write a simple front that shows progress or gives the user information that the image is being generated
# 4. Present the result for user in simple front, maybe add download button so user can download the new image



if __name__ == "__main__":
        
    app.run(debug=True)