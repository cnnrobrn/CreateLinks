import streamlit as st
from PIL import Image
from clarifai.client.model import Model

# Your PAT (Personal Access Token) can be found in the Account's Security section
# USER_ID = "clarifai"
# APP_ID = "main"

# Set the model using the model URL or model ID
model_url = "https://clarifai.com/clarifai/main/models/apparel-detection"
detector_model = Model(
    url=model_url,
    pat="63da8300e0fc46268946b6c7d88cd6fe",
)

# Streamlit file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Only proceed if the user has uploaded a file
if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Read the uploaded image as bytes
    image_bytes = uploaded_file.read()

    # Use the Clarifai model to make a prediction with the image bytes
    prediction_response = detector_model.predict_by_bytes(image_bytes, input_type="image")

    # Check if there are any regions in the prediction output
    if prediction_response.outputs:
        regions = prediction_response.outputs[0].data.regions

        # Loop through the prediction results
        for region in regions:
            # Accessing and rounding the bounding box values
            top_row = round(region.region_info.bounding_box.top_row, 3)
            left_col = round(region.region_info.bounding_box.left_col, 3)
            bottom_row = round(region.region_info.bounding_box.bottom_row, 3)
            right_col = round(region.region_info.bounding_box.right_col, 3)

            for concept in region.data.concepts:
                # Accessing and rounding the concept value
                name = concept.name
                value = round(concept.value, 4)

                # Display the prediction results in Streamlit
                st.write(f"{name}: {value} BBox: {top_row}, {left_col}, {bottom_row}, {right_col}")
    else:
        st.write("No regions found in the image.")
else:
    st.write("Please upload an image file.")
