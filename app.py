import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from werkzeug.utils import secure_filename
import numpy as np
from skimage.color import rgb2gray


ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
IMAGE_SIZE = (48, 48)
UPLOAD_FOLDER = 'uploads'
modres = load_model('lenet_weights1.hdf5')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def predict(file):
	img  = load_img(file, target_size=IMAGE_SIZE)
	img = img_to_array(img)/255.0
	img = rgb2gray(img)
	img = np.expand_dims(img, axis=0)
	probs = modres.predict(img)
	a = np.argmax(probs)
	a_max = np.max(probs)
	prob_temp = probs.flatten()
	prob_temp[a] = 0.0
	b = np.argmax(prob_temp)
	b_max = np.max(prob_temp)
	#probs[a] = -1
	#b = np.argmax(probs)
	#b_max = np.max(probs)
	dicti = {0:'angry', 1:'irritated', 2:'fearful', 3:'happy', 4:'sad', 5:'surprised', 6:'neutral'}
	#output = {dicti.get(a): a_max}
	output = []
	output.append(dicti.get(a))
	output.append(a_max)
	output.append(dicti.get(b))
	output.append(b_max)
	return output

app = Flask(__name__, template_folder='Templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template('home.html', label='', imagesource='file://null')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            output = predict(file_path)
    return render_template("home.html", label=output, imagesource=file_path)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(threaded=False)