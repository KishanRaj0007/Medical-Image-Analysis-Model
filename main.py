import streamlit as st
from keras.models import load_model
from PIL import Image
from util import classify, set_background

set_background('dsa.jpg')

# Set title
st.markdown('<h1 style="color: black;">Disease Detection Model</h1>', unsafe_allow_html=True)

# Let the user choose which disease to detect
disease = st.radio("Select Disease to Detect", ["Lung Cancer", "Alzheimer's Disease", "Pneumonia"])

st.markdown('<h3 style="color: black;">Upload brain MRI image for Alzheimer, chest X ray of Pneumonia and Histopathological image for lung cancer</h3>', unsafe_allow_html=True)

# Set header with smaller font size and light color
st.markdown('<h2 style="color: black;">Please upload the Medical image of the patient</h2>', unsafe_allow_html=True)

# Upload file
file = st.file_uploader('', type=['jpeg', 'png', 'jpg'])

# Load the classifier based on the selected disease
if disease == "Alzheimer's Disease":
    model_path = 'keras_model1.h5'
    labels_path = 'labels1.txt'
elif disease == "Lung Cancer":
    model_path = 'keras_model2.h5'
    labels_path = 'labels2.txt'
elif disease == "Pneumonia":
    model_path = 'keras_model3.h5'
    labels_path = 'labels3.txt'


model = load_model(model_path)

# Load the class name

with open(labels_path, 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()
print(class_names)

# Display image which user has selected
if file is not None:
    image = Image.open(file).convert('RGB')
    resized_image = image.resize((224, 224))
    st.image(resized_image, use_column_width=True)

    # Classify image
    class_name, conf_score = classify(resized_image, model, class_names, disease)

    # Write classification
    st.markdown(f'<h3 style="color: #FF6347;">Class: {class_name}</h3>', unsafe_allow_html=True)
    st.markdown(f'<h4 style="color: #A0522D;">Confidence Score: {conf_score}</h4>', unsafe_allow_html=True)



