from flask import Flask
from flask import render_template
app = Flask(__name__)

# TASKS:
# 1. Write function that loads the model and prepares it on server launch
# 1.1 Make it so it only loads on >>server launch<< not on every user connection if you know how
# 2. Write simple homepage that let's you upload an content and style images
# 3. Write a function that uses style_transfer() function from style_transfer.py to generate new images from uploaded content/style images
# 3.1 Write a simple front that shows progress or gives the user information that the image is being generated
# 4. Present the result for user in simple front, maybe add download button so user can download the new image

@app.route('/')
def hello():
    return render_template('hello.html');