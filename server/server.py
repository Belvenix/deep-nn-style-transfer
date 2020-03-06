from flask import Flask, render_template, request, redirect, abort, session, url_for
from werkzeug.utils import secure_filename

from redis import Redis
import rq

from uuid import uuid4
import os

from style_transfer import style_transfer_wrapper, RESULTS_PATH, IMAGES_PATH

app = Flask(__name__)


# CONSTANTS
RQ_WORKER_NAME = "transfer-task"
REDIS_URL = "redis://"
ALLOWED_TYPES = ["image/jpeg"]


# TASKS:
# 1. Write function that loads the model and prepares it on server launch
# 1.1 Make it so it only loads on >>server launch<< not on every user connection if you know how


# 2. Write simple homepage that let's you upload an content and style images
app.secret_key = os.urandom(16)
queue = rq.Queue(RQ_WORKER_NAME, connection=Redis.from_url('redis://'))


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

    uid = uuid4()
    session['user_id'] = uid
    style_image_name = str(uid) + "_style.jpg"
    content_image_name = str(uid) + "_content.jpg"
    output_image_name = str(uid) + "_out.jpg"

    style_image.save(IMAGES_PATH + style_image_name)
    content_image.save(IMAGES_PATH + content_image_name)

    job = queue.enqueue(style_transfer_wrapper, style_image_name, content_image_name, output_image_name)
    session['job_id'] = job.id
    job.meta['progress'] = 0
    job.save_meta()

    return redirect(url_for('get_status'))


@app.route('/status')
def get_status():
    job = queue.fetch_job(session['job_id'])
    if job is None:
        return "no job"
    progress = job.meta['progress']
    if progress is not None:
        return str(progress)
    else:
        return "sth went wrong"

# 3. Write a function that uses style_transfer() function from style_transfer.py to generate new images
# from uploaded content/style images
# 3.1 Write a simple front that shows progress or gives the user information that the image is being generated
# 4. Present the result for user in simple front, maybe add download button so user can download the new image


if __name__ == "__main__":
        
    app.run(debug=True)
