import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import cv2

# Streamlit UI setup
st.set_page_config(page_title="Breast Cancer Detector", layout="centered")
st.title("🩺 Breast Ultrasound Image Classifier")
st.write("Upload a breast ultrasound image to detect if it is **normal**, **benign**, or **malignant**.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# Model selector
model_option = st.selectbox(
    "Choose model for prediction:",
    ["MobileNetV2 CNN", "SVM"]
)

# Class names
class_names = ['benign', 'malignant', 'normal']

# Load CNN model for both direct prediction and feature extraction
cnn_model = tf.keras.models.load_model("mobilenetv2_model.keras")

# Load SVM model
svm_model = joblib.load("svm_classifier.pkl")

if uploaded_file is not None:
    # Preprocess image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224)) / 255.0
    input_img = np.expand_dims(img_resized, axis=0)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    if model_option == "MobileNetV2 CNN":
        cnn_pred = cnn_model.predict(input_img)
        label = np.argmax(cnn_pred)
        st.subheader("MobileNetV2 CNN Prediction:")
        st.success(f"{class_names[label]}")

    elif model_option == "SVM":
        # Extract features from CNN before the final classifier
        feature_extractor = tf.keras.models.Model(
            inputs=cnn_model.input,
            outputs=cnn_model.get_layer(index=-3).output  # Adjust if necessary
        )
        features = feature_extractor.predict(input_img)
        label = svm_model.predict(features)[0]
        st.subheader("SVM Prediction:")
        st.success(f"{class_names[label]}")
