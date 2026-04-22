from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ================= STARTUP =================
print("\n🔄 Starting backend initialization...")

# Load Yield Model
try:
    yield_model = joblib.load("models/yield_model.pkl")
    print("✅ Yield model loaded")
except Exception as e:
    print("❌ Yield model failed:", e)
    exit()

# Load Disease Model
try:
    disease_model = load_model("models/disease_model.h5")
    print("✅ Disease model loaded")
except Exception as e:
    print("❌ Disease model failed:", e)
    exit()

# Class Labels
class_names = [
    "Potato Early Blight",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Healthy"
]

print("🚀 Backend initialization complete\n")


# ================= ROUTES =================

@app.route("/")
def home():
    return "Backend running ✅"


# -------- YIELD PREDICTION --------
@app.route("/predict-yield", methods=["POST"])
def predict_yield():
    try:
        data = request.get_json()

        rainfall = float(data["rainfall"])
        pesticides = float(data["pesticides"])
        temperature = float(data["temperature"])

        features = np.array([[rainfall, pesticides, temperature]])
        prediction = yield_model.predict(features)

        return jsonify({
            "predicted_yield": float(prediction[0])
        })

    except Exception as e:
        print("❌ Yield Error:", e)
        return jsonify({"error": str(e)}), 400


# -------- DISEASE PREDICTION --------
@app.route("/predict-disease", methods=["POST"])
def predict_disease():
    try:
        # Check file exists
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']

        # Process image
        img = Image.open(file).convert("RGB").resize((224, 224))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = disease_model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        return jsonify({
            "disease": predicted_class
        })

    except Exception as e:
        print("❌ Disease Error:", e)   # 🔥 CRITICAL DEBUG LINE
        return jsonify({"error": str(e)}), 500


# ================= RUN =================
if __name__ == "__main__":
    print("🌐 Backend running at: http://127.0.0.1:5001\n")

    app.run(host='0.0.0.0', port=5001, debug=False)