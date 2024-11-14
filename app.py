import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model using a relative path (ensure the model is in the same folder as the app)
model = load_model('ecg_model.h5')  # This will look for the model in the current working directory

# Define the image size used during training (ensure it matches your training process)
IMG_SIZE = 150

# Streamlit title and description
st.title("Heartbeat Abnormality Prediction App")
st.write("This app predicts whether a heartbeat is normal or abnormal based on the uploaded image.")

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to preprocess and predict
def prepare_image(image_file):
    img = image.load_img(image_file, target_size=(IMG_SIZE, IMG_SIZE))  # Resize to match model input
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale the image as we did during training
    return img_array

# When the user uploads an image, predict and show the result
if uploaded_image is not None:
    # Prepare the image
    img_array = prepare_image(uploaded_image)

    # Predict the class (normal or abnormal)
    prediction = model.predict(img_array)
    
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    
    # Show the prediction result
    if prediction[0] > 0.5:
        st.write("Prediction: Abnormal Heartbeat")
    else:
        st.write("Prediction: Normal Heartbeat")
