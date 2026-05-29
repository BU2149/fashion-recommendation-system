# 👗 StyleSnap AI – Fashion Recommendation System

An AI-powered Fashion Recommendation System that recommends visually similar outfits using Deep Learning and Computer Vision.

## ✨ Features

- 📸 Upload fashion image
- 🤖 AI-based recommendation engine
- 🧠 Deep learning feature extraction using ResNet50
- 🎯 Similar outfit recommendation using KNN
- 🌈 Stylish Streamlit UI
- ⚡ Fast image similarity matching

## 🛠️ Tech Stack

- Python
- TensorFlow
- ResNet50
- Scikit-learn (KNN)
- Streamlit
- NumPy
- Pickle

## 📂 Project Structure

```bash
fashion-recommendation-system/
│── app.py
│── main.py
│── embeddings.pkl
│── filenames.pkl
│── images.csv
│── images/
│── uploads/
│── README.md
```

## 🚀 How to Run

Clone repository:

```bash
git clone https://github.com/BU2149/fashion-recommendation-system.git
```

Go inside folder:

```bash
cd fashion-recommendation-system
```

Install requirements:

```bash
pip install -r requirements.txt
```

Run app:

```bash
streamlit run main.py
```

## 🧠 Model Used

This project uses **ResNet50 (ImageNet pre-trained)** for feature extraction and **KNN** for finding visually similar fashion products.

## 🎯 Future Improvements

- Outfit category prediction
- Personalized recommendations
- Fashion trend detection
- Better UI animations

## 👩‍💻 Author

**Navya Juneja**
