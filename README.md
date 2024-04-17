# Description
The project is a Disease Detection Model, 
which allows users to upload medical images of patients 
and detect various diseases such as Lung Cancer, 
Alzheimer's Disease, and Pneumonia. It utilizes deep 
learning techniques to analyze the images and classify them 
into different disease categories based on pre-trained 
models. The application aims to assist healthcare 
professionals in diagnosing diseases accurately and 
efficiently. 


# Features
Disease Detection: Users can select the type of disease they 
want to detect from a list including Lung Cancer, Alzheimer's Disease, and 
Pneumonia. 

 ### Image Upload: 
 Users can upload medical images of patients in common 
formats such as JPEG, PNG, and JPG. 

 ### Classification: 
 The uploaded images are classified using pre-trained deep 
learning models specific to each disease category. 

 ### Confidence Score: 
 The application provides a confidence score along with the 
predicted disease class, indicating the reliability of the prediction. 

 ### Customizable Background: 
 The user interface offers a customizable background, 
enhancing the overall visual appeal and user experience.


# Tech Stack 
Python,  Streamlit, Tensorflow, Keras, Pillow, 
Numpy, Base64, Html/css, 
h5py,importlib_metadata,jax,pandas,anaconda,pip,scikit
learn,tensorboard,simpleitk,tensorboardx,torchaudio,torchmetr
 ics,torchvision, pytorch(during learning in jupyter 
notebook),pydicom, opencv, matplotlib(visualization in jupyter), 
cuda(for activating gpu),etc..



# Methodology
A systematic approach to data collection and 
preprocessing is adopted. A diverse dataset of medical images 
representing each disease category is gathered. These images 
undergo preprocessing steps to standardize their size, format, 
and quality, ensuring consistency and compatibility for model 
training and evaluation. 
The core of the project lies in model development, where pre
trained deep learning models are utilized for image classification 
tasks. 
The efficiency of the model is subsequently increased because we 
are not required to train the model each time we predict the 
disease. Moreover due to lack of large number of dataset a great 
focus is kept on training the model. First I have tried to train the 
model with 50 epochs, then by observing the confusion matrix 
and loss graph of validation and training sets of images, 
subsequently epochs was decreased so that machine do not 
overfit on the data. Also batch size was decreased to maintain the 
accuracy which might have compromised in decreasing the 
epochs. 
Transfer learning techniques are applied to fine-tune these 
models, adapting them to the specific requirements of disease
detection. Separate models are trained for each disease category, 
optimizing performance and accuracy while addressing the 
unique characteristics of each condition. 
Following model development, the project shifts focus to 
application development. An interactive web application is built 
using Streamlit, a Python library tailored for data science 
applications. The application features intuitive user interfaces, 
allowing users to upload medical images, select the disease for 
detection, and receive real-time predictions along with 
confidence scores. 



# Resources
1. Udemy course:-  Deep Learning with Pytorch for Medical 
Image Analysis. 
2. StackOverFlow and Github for problem solutions 
3. Youtube for Pycharm installations and setup and for 
creating a conda environment to enable GPU 
4. Kaggle for datasets 
5. Anaconda documentation for change in syntax due to 
version updation. 
6. Teachable machine website for training the dataset 
7. Streamlit and keras documentation for syntax to make a 
Web App 
8. Reddit for problem solutions 
9. ChatGPT for detecting possible solutions for a critical stuck 
situation.
10. Pytorch tutorial(torchvision and torchmetrics included)
11. Pydicom documentation
12. pytorch lightening documentation(https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) -  MOST HELPFUL
