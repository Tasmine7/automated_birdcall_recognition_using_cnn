from flask import Flask, render_template, request, redirect, url_for, session
import os
import uuid
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # non-GUI backend for matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from werkzeug.utils import secure_filename

# ---------------- CONFIG ----------------
app = Flask(__name__)
# CHANGE THIS before deploying to production
app.secret_key = "change_this_to_a_random_secret_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
SPECTRO_FOLDER = os.path.join(BASE_DIR, "static", "spectrograms")
IMAGE_FOLDER = "images"  # under static/images/

# ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SPECTRO_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SPECTRO_FOLDER"] = SPECTRO_FOLDER

# ---------------- MODEL ----------------
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
# load model once at startup
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------- CLASS MAP (edit images/names as needed) ----------------
CLASS_MAP = {
    0: {
        "name": "American Robin",
        "image": "american_robin.jpg",
        "info": "American Robin:The American Robin is one of the most familiar birds across North America, found from Alaska and Canada through the United States and into Mexico. It thrives in woodlands, parks, and suburban lawns, where it’s often seen hunting earthworms. This species is classified as Least Concern, with a large, stable population and no major threats. Robins play an important role in controlling insects and dispersing seeds, making them a key part of their ecosystems."

    },
    1: {
        "name": "Bewick's Wren",
        "image": "bewicks_wren.jpg",
        "info": "Bewick’s Wren:Bewick’s Wren is native to western North America, especially along the Pacific Coast and the Southwest. While the species remains stable in these regions, it has disappeared from much of the eastern United States due to habitat changes and competition from other wrens. Overall, it is listed as Least Concern, but some populations are of conservation interest. Bewick’s Wrens are active, agile insect hunters known for their loud, expressive songs"
 

    },
    2: {
        "name": "Northern Cardinal",
        "image": "northern_cardinal.jpg",
        "info":  "Northern Cardinal:The Northern Cardinal is a striking and widely recognized species found across eastern and central North America. It lives in woodlands, gardens, and backyards, with the bright red males making it one of the continent’s most iconic birds. The species has a strong and growing population and is considered Least Concern. Cardinals are highly territorial and often visit feeders, making them a common sight in residential areas."


          },
    3: {
        "name": "Northern Mockingbird",
        "image": "northern_mockingbird.jpg",
        "info": "Northern Mockingbird:The Northern Mockingbird is known for its incredible ability to imitate other bird calls, animal sounds, and even mechanical noises. It lives year-round in much of the United States, Mexico, and southern Canada, preferring open habitats with shrubs and scattered trees. Its conservation status is Least Concern, with stable, healthy populations. This species is a confident, vocal presence in many neighborhoods, often singing throughout the day — and sometimes at night."


    },
    4: {
        "name": "Song Sparrow",
        "image": "song_sparrow.jpg",
        "info":"Song Sparrow:The Song Sparrow is one of the most widespread and adaptable sparrow species in North America. It inhabits marshes, fields, shrublands, and suburban landscapes, with regional variations in plumage and song. The species is abundant and listed as Least Concern, though a few local subspecies face habitat pressure. Known for its rich, melodious singing, the Song Sparrow plays an important role in insect control and seed dispersal."
    }
}

# ---------------- AUDIO PREPROCESSING ----------------
def audio_to_mfcc_vector(audio_path, n_mfcc=40):
    """
    Loads ANY audio format (wav, mp3, ogg, flac, etc.)
    Converts to mono, 22050 Hz, extracts MFCCs,
    averages over time → returns shape (1, 40, 1)
    """

    # Load using librosa (works for mp3, wav, ogg, flac…)
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    # Handle edge case: silent/empty audio
    if y is None or len(y) == 0:
        raise ValueError("Uploaded audio contains no valid samples.")

    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Mean over time axis → (40,)
    mfcc_mean = np.mean(mfccs, axis=1).astype("float32")

    # Match model input (1, 40, 1)
    mfcc_input = np.expand_dims(mfcc_mean, axis=0)   # (1, 40)
    mfcc_input = np.expand_dims(mfcc_input, axis=-1) # (1, 40, 1)

    return mfcc_input


def create_mel_spectrogram(audio_path, save_folder):
    """
    Create mel spectrogram PNG and return filename (saved under save_folder).
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(8, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.tight_layout()

    filename = f"spec_{uuid.uuid4().hex}.png"
    filepath = os.path.join(save_folder, filename)
    fig.savefig(filepath, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    return filename

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # ensure file exists in request
        if "audio_file" not in request.files:
            return redirect(request.url)

        file = request.files["audio_file"]
        if file.filename == "":
            return redirect(request.url)

        # save file
        fname = secure_filename(file.filename)
        audio_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        file.save(audio_path)

        # preprocessing & predict
        mfcc_input = audio_to_mfcc_vector(audio_path)
        preds = model.predict(mfcc_input)
        pred_index = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds)) * 100.0

        # map to species & image
        species_info = CLASS_MAP.get(pred_index, {"name": f"Unknown ({pred_index})", "image": None})
        species_name = species_info["name"]
        image_file = species_info["image"]
        bird_info = species_info.get("info", "No additional information available.")

        bird_image_url = None
        if image_file:
            bird_image_url = url_for("static", filename=f"{IMAGE_FOLDER}/{image_file}")

        # create spectrogram
        spec_filename = create_mel_spectrogram(audio_path, app.config["SPECTRO_FOLDER"])
        spectrogram_url = url_for("static", filename=f"spectrograms/{spec_filename}")

        # store result once in session then redirect with no_splash flag
        session["prediction_data"] = {
            "prediction": species_name,
            "confidence": round(confidence, 2),
            "spectrogram_url": spectrogram_url,
            "bird_image_url": bird_image_url,
            "audio_filename": os.path.basename(audio_path),
             "bird_info": bird_info,
        }

        return redirect(url_for("index", no_splash=1))

    # GET: pop prediction (so refresh does not keep old results)
    data = session.pop("prediction_data", None)
    if data is None:
        data = {
            "prediction": None,
            "confidence": None,
            "spectrogram_url": None,
            "bird_image_url": None,
            "audio_filename": None,
            "bird_info": None

        }

    return render_template("index.html", **data)

if __name__ == "__main__":
    # debug True only while developing locally
    app.run(host="0.0.0.0", port=5000, debug=True)
