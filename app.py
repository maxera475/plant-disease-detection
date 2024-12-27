import streamlit as st
import requests
from PIL import Image
import io

# Define FastAPI server URL
FASTAPI_URL = "http://localhost:8000/predict"

# Streamlit app title and description
st.title("Plant Disease Detection")
st.write("""
Upload an image of a plant leaf, and the model will predict the disease along with its confidence score.
""")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prediction button
    if st.button("Predict"):
        with st.spinner("Processing..."):
            try:
                # Convert image to byte array
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="PNG")
                img_byte_arr = img_byte_arr.getvalue()

                # Send POST request to FastAPI
                response = requests.post(
                    FASTAPI_URL, 
                    files={"file": ("filename", img_byte_arr, "image/png")}
                )

                # Handle response
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Prediction: {result['class']}")
                    st.info(f"Confidence: {result['confidence']:.2f}")
                else:
                    st.error("Error occurred while making prediction. Please try again.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
