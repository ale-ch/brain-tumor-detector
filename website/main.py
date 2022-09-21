from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename

import os
import tensorflow as tf
import numpy as np

UPLOAD_FOLDER = os.path.join(os.getcwd(), "website", "static", "uploads")

app = Flask(__name__)
app.secret_key = "34y5oj3n4b53gyvuh"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

model_path = os.path.join(os.getcwd(), "detector", "model")
model = tf.keras.models.load_model(model_path)
class_names = ['brain tumor', 'healthy']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = tf.keras.utils.load_img(image_path, target_size=(180,180))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    pred = model.predict(img_array)
    score = tf.nn.sigmoid(pred[0])
    classification = f"Classification: {class_names[np.argmax(score)]} with {100 * np.max(score):.2f} percent confidence."

    return classification

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/", methods=['POST'])
def upload_image():
    if 'imagefile' not in request.files:
        flash('No file part', category='error')
        return redirect(request.url)
    imagefile = request.files['imagefile']
    if imagefile.filename == '':
        flash("Please select an image.", category='error')
        return redirect(request.url)
    if imagefile and allowed_file(imagefile.filename):
        filename = secure_filename(imagefile.filename)
        imagefile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        classification = predict(filename)    
        flash("Image uploaded successfully!", category='success')
        return render_template("index.html", filename=filename, prediction=classification) 
    else:
        flash("Allowed image formats are: png, jpg, jpeg, gif", category='error')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename="uploads/" + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)
