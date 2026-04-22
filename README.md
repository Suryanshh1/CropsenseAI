# 🌾 CropSense AI – Smart Farming Intelligence Platform

🌍 **Live Demo:** [https://cropsense-suryansh.surge.sh](https://cropsense-suryansh.surge.sh)  
⚙️ **Backend API:** [https://huggingface.co/spaces/Suryanshh1/cropsense-backend](https://huggingface.co/spaces/Suryanshh1/cropsense-backend)

CropSense AI is an AI-based web application that helps farmers with:
- 🌿 Crop Disease Detection (using Deep Learning)
- 🌾 Yield Prediction (using Machine Learning)

## 🚀 Features
- Upload leaf image → detect disease
- Enter environmental data → predict crop yield
- Real-time results using Flask API

## 🧠 Tech Stack
- **Backend:** Python, Flask, TensorFlow / Keras, Scikit-learn
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** 
  - Frontend hosted on **Surge**
  - Backend API hosted on **Hugging Face Spaces** (Docker)

## 📂 Project Structure
- `backend/` → Flask API & models, including `Dockerfile` for Hugging Face
- `frontend/` → UI, including `index.html` configured for production

## ▶️ Run Locally

### Backend
1. Navigate to the `backend` folder: `cd backend`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Flask server: `python app.py`

### Frontend
1. Open `frontend/index.html` in any modern web browser.
2. Ensure you change `API_BASE_URL` inside `index.html` back to `http://127.0.0.1:5001` if you wish to run the backend locally instead of using the cloud API.
