import streamlit as st
from PIL import Image
from clarifai.client.model import Model

CLARIFAI_PAT="63da8300e0fc46268946b6c7d88cd6fe"

# Set the model URL and your Personal Access Token
model_url = "https://clarifai.com/clarifai/main/models/apparel-detection"
detector_model = Model(
    url=model_url,
    pat="63da8300e0fc46268946b6c7d88cd6fe",
)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Define the output file path
    output_path = f"./Uploadedfile/{uploaded_file.name}"

    # Write the file to the output path
    with open(output_path, "wb") as output_file:
        output_file.write(uploaded_file.getbuffer())

    # Confirmation
    st.success(f"File saved as {output_path}")


    if output_path is not None:

        # Convert uploaded file to bytes
        image_bytes = uploaded_file.read()
        #st.write("Bytes: ",len(image_bytes),": ",image_bytes)

        try:
            # Make the prediction using Clarifai
            prediction_response = Model(model_url).predict_by_filepath(output_path, input_type="image")
            st.write(f"Clarifai response status: {prediction_response.status}")
            
            if prediction_response.outputs:
                regions = prediction_response.outputs[0].data.regions
                for region in regions:
                    top_row = round(region.region_info.bounding_box.top_row, 3)
                    left_col = round(region.region_info.bounding_box.left_col, 3)
                    bottom_row = round(region.region_info.bounding_box.bottom_row, 3)
                    right_col = round(region.region_info.bounding_box.right_col, 3)
                    for concept in region.data.concepts:
                        name = concept.name
                        value = round(concept.value, 4)
                        st.write(f"{name}: {value} BBox: {top_row}, {left_col}, {bottom_row}, {right_col}")
            else:
                st.write("No regions found in the image.")

        except Exception as e:
            # Print the full exception error
            st.write(f"Error occurred: {str(e)}")
