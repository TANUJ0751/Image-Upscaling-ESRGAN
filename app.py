import os.path as osp
import glob

import numpy as np
import torch
import sys
import streamlit as st
import cv2
from PIL import Image
sys.path.append('./Real-ESRGAN')
import RRDBNet_arch as arch
device = torch.device('cuda')
st.write("Image Upscaler By tanuj")
uploaded_image=st.file_uploader("Upload Image file",["JPG","JPEG","PNG"])
if uploaded_image is not None:
    image=Image.open(uploaded_image)
    st.image(image)
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = img.astype(np.float32) * 1.0 / 255.0
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)
