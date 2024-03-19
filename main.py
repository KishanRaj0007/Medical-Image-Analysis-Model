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


#classifier for Alzheimer's disease

# from keras.models import load_model  # TensorFlow is required for Keras to work
# from PIL import Image, ImageOps  # Install pillow instead of PIL
# import numpy as np

# Disable scientific notation for clarity
# np.set_printoptions(suppress=True)

# Load the model
# model = load_model("keras_Model.h5", compile=False)

# Load the labels
# class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
# data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
# image = Image.open("<IMAGE_PATH>").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
# size = (224, 224)
# image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
# image_array = np.asarray(image)

# Normalize the image
# normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
# data[0] = normalized_image_array

# Predicts the model
# prediction = model.predict(data)
# index = np.argmax(prediction)
# class_name = class_names[index]
# confidence_score = prediction[0][index]

# Print prediction and confidence score
# print("Class:", class_name[2:], end="")
# print("Confidence Score:", confidence_score)





#Classifier for lung cancer

# from keras.models import load_model  # TensorFlow is required for Keras to work
# from PIL import Image, ImageOps  # Install pillow instead of PIL
# import numpy as np
#
# # Disable scientific notation for clarity
# np.set_printoptions(suppress=True)
#
# # Load the model
# model = load_model("keras_Model.h5", compile=False)
#
# # Load the labels
# class_names = open("labels.txt", "r").readlines()
#
# # Create the array of the right shape to feed into the keras model
# # The 'length' or number of images you can put into the array is
# # determined by the first position in the shape tuple, in this case 1
# data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
#
# # Replace this with the path to your image
# image = Image.open("<IMAGE_PATH>").convert("RGB")
#
# # resizing the image to be at least 224x224 and then cropping from the center
# size = (224, 224)
# image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
#
# # turn the image into a numpy array
# image_array = np.asarray(image)
#
# # Normalize the image
# normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
#
# # Load the image into the array
# data[0] = normalized_image_array
#
# # Predicts the model
# prediction = model.predict(data)
# index = np.argmax(prediction)
# class_name = class_names[index]
# confidence_score = prediction[0][index]

# # Print prediction and confidence score
# print("Class:", class_name[2:], end="")
# print("Confidence Score:", confidence_score)


