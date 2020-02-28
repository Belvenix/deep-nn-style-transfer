from flask import Flask, flash, url_for, render_template

import torch
import torchvision.models as models

app = Flask(__name__)

SESSION_TYPE = 'filesystem'
UPLOAD_FOLDER = '/var/www/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# TASKS:
# 1. Write function that loads the model and prepares it on server launch
# 1.1 Make it so it only loads on >>server launch<< not on every user connection if you know how

def initialize(app):
    app.mean = [0.485, 0.456, 0.406]
    app.std = [0.229, 0.224, 0.225]
    app.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    app.vgg19_model = models.vgg19(pretrained=True).features.to(app.device).eval()
    app.imsize = (512,512) if torch.cuda.is_available() else (300,300)

# 2. Write simple homepage that let's you upload an content and style images

@app.route('/', methods=['POST','GET'])
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    pass

# 3. Write a function that uses style_transfer() function from style_transfer.py to generate new images from uploaded content/style images
# 3.1 Write a simple front that shows progress or gives the user information that the image is being generated
# 4. Present the result for user in simple front, maybe add download button so user can download the new image



if __name__ == "__main__":
    
    initialize(app)    
    app.run(debug=True)