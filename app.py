import os
import cv2
import numpy as np
import torch
import streamlit as st
from pathlib import Path
from PIL import Image

# Append the path for the model architecture file
import sys
sys.path.append('./Real-ESRGAN')
import RRDBNet_arch as arch

# Load model
@st.cache_resource
def load_model():
    model_path = 'RRDB_ESRGAN_x4.pth'
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()
    return model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Upscaling function
def upscale_image(uploaded_image, model, device):
    # Convert the uploaded image to OpenCV format
    img = np.array(Image.open(uploaded_image))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Normalize and prepare the image for the model
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    # Convert output to image format
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    return output

# Streamlit UI
st.title("Image Upscaling with ESRGAN")
st.write("Upload a low-resolution image, and the app will upscale it using ESRGAN.")

uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.write("Processing...")
    model = load_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Upscale the uploaded image
    upscaled_image = upscale_image(uploaded_file, model, device)

    # Display results
    st.image(uploaded_file, caption="Original Image", use_container_width=True)
    st.image(upscaled_image, caption="Upscaled Image", use_container_width=True)

    # Option to download the upscaled image
    output_path = Path("upscaled_image.png")
    cv2.imwrite(str(output_path), cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))
    with open(output_path, "rb") as file:
        st.download_button(
            label="Download Upscaled Image",
            data=file,
            file_name="upscaled_image.png",
            mime="image/png"
        )
