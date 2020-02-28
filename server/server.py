from flask import Flask, flash, request, redirect, url_for, render_template, session
from werkzeug.utils import secure_filename
from flask_session import Session
from flask import send_from_directory

import torch
import torchvision.models as models

app = Flask(__name__)
sess = Session()

SESSION_TYPE = 'filesystem'
UPLOAD_FOLDER = '/var/www/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}



def initialize(app):
    app.mean = [0.485, 0.456, 0.406]
    app.std = [0.229, 0.224, 0.225]
    app.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    app.vgg19_model = models.vgg19(pretrained=True).features.to(app.device).eval()
    app.imsize = (512,512) if torch.cuda.is_available() else (300,300)
    
    #For session to work we need a key
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['SESSION_TYPE'] = SESSION_TYPE
    
    #Debug print to check whether we initialize it once
    print(app.vgg19_model)
    sess.init_app(app)

# TASKS:
# 1. Write function that loads the model and prepares it on server launch
# 1.1 Make it so it only loads on >>server launch<< not on every user connection if you know how
# 2. Write simple homepage that let's you upload an content and style images
# 3. Write a function that uses style_transfer() function from style_transfer.py to generate new images from uploaded content/style images
# 3.1 Write a simple front that shows progress or gives the user information that the image is being generated
# 4. Present the result for user in simple front, maybe add download button so user can download the new image

@app.route('/xyz', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        user_image = request.form['myfile']
        print(type(user_image))
        #return 'File was submitted. Please wait patiently.'
    else:
        pass
        #return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'myfile' not in request.files:
            flash('No file part')
            return redirect(request.url)
        f = request.files['myfile']
        if f.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            print(filename)
            f.save(app.config['UPLOAD_FOLDER'] + secure_filename(f.filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    
    initialize(app)    
    app.run(debug=True)