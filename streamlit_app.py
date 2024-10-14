import streamlit as st
import os
from pathlib import Path

# Set the title of the app
st.title("Image Uploader")

from PIL import Image

# Add a file uploader to the app
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # If the user uploads a file, open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Image uploaded successfully!")
else:
    st.write("Please upload an image file.")