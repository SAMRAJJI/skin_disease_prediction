from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
import numpy as np
import cv2
import pickle
import os

app = Flask(__name__)

# Load model and LabelEncoder
model = tf.keras.models.load_model("skin_cnn.keras")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

IMG_SIZE = 64  # same as training

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == "":
        return redirect(request.url)
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess image
    img = cv2.imread(filepath)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)
    disease = le.inverse_transform([np.argmax(pred)])[0]

    return render_template("index.html", filename=file.filename, prediction=disease)

# Show uploaded image
@app.route("/display/<filename>")
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)
