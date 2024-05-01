import streamlit as st

from streamlit_option_menu import option_menu
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
from FruitModel import FruitModel



# sidebar
with st.sidebar:
    selected = option_menu('Industrial Project', [
        'Fruit Classification',
        # 'Second Prediction',
    ],
        icons=['','activity',],
        default_index=0)


#Fruit Classification
if selected == 'Fruit Classification': 

    fruit_model = FruitModel()

    # Title
    st.write('# Fruit Classification using CNN')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image',  width=300)

        if st.button('Predict'):
            prediction = fruit_model.make_predictions(uploaded_file)
            if prediction > 0.5:
                st.error('Prediction: Rotten Fruit ')
            else:
                st.success('Prediction: Fresh Fruit ')
