import base64
import numpy as np
import streamlit as st
from PIL import ImageOps,Image

def set_background(image_file):
    with open(image_file,"rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
             background-image: url(data:image/jpg;base64,{b64_encoded});
             background-size: cover;
        }}
        </style>
        """
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model, class_names, disease):


    #convert image to (224,224)
    image = ImageOps.fit(image,(224,224), Image.Resampling.LANCZOS)

    #convert image to numpy array
    image_array = np.asarray(image)

    #normalize image
    normalized_image_array = (image_array.astype(np.float32)/127.5) - 1

    #set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    data[0] = normalized_image_array

    #make prediction
    prediction = model.predict(data)
    if disease == "Pneumonia":
        index = 0 if prediction[0][0] > 0.95 else 1
    else:
        index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]


    return class_name, confidence_score