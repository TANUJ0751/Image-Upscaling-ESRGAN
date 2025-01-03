import torch
from PIL import Image
import numpy as np
import streamlit as st

from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

path_to_image = 'inputs/lr_image.png'
img = Image.open(path_to_image)
st.image(img)
image = Image.open(path_to_image).convert('RGB')

sr_image = model.predict(image)
st.image(sr_image)
sr_image.save('results/sr_image.png')
