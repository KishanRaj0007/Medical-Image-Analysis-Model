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


#Preprocess
# from pathlib import Path
# import pydicom
# import numpy as np
# import cv2
# import pandas as pd
# import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm #to get a progress bar
# labels = pd.read_csv(r"C:\Users\kisha\OneDrive\Desktop\Projects\MedAI\AI-IN-MEDICAL-MATERIALS\04-Pneumonia-Classification\stage_2_train_labels.csv")
# labels.head(6)
# patientId x y width height Target
# 0 0004cfab-14fd-4e49-80ba-63a80b6bddd6 NaN NaN NaN NaN 0
# 1 00313ee0-9eaa-42f4-b0ab-c148ed3241cd NaN NaN NaN NaN 0
# 2 00322d4d-1c29-4943-afc9-b6754be640eb NaN NaN NaN NaN 0
# 3 003d8fa0-6bf1-40ed-b54c-ac657f8495c5 NaN NaN NaN NaN 0
# 4 00436515-870c-4b36-a041-de91049b9ab4 264.0 152.0 213.0 379.0 1
# 5 00436515-870c-4b36-a041-de91049b9ab4 562.0 152.0 256.0 453.0 1
# labels = labels.drop_duplicates("patientId")
# labels.head()
# patientId x y width height Target
# 0 0004cfab-14fd-4e49-80ba-63a80b6bddd6 NaN NaN NaN NaN 0
# 1 00313ee0-9eaa-42f4-b0ab-c148ed3241cd NaN NaN NaN NaN 0
# 2 00322d4d-1c29-4943-afc9-b6754be640eb NaN NaN NaN NaN 0
# 3 003d8fa0-6bf1-40ed-b54c-ac657f8495c5 NaN NaN NaN NaN 0
# 4 00436515-870c-4b36-a041-de91049b9ab4 264.0 152.0 213.0 379.0 1
# ROOT_PATH = Path(r"C:\Users\kisha\OneDrive\Desktop\Projects\MedAI\AI-IN-MEDICAL-MATERIALS\04-Pneumonia-Classification/stage_2_train_images/")
# SAVE_PATH = Path(r"C:\Users\kisha\OneDrive\Desktop\Projects\MedAI\AI-IN-MEDICAL-MATERIALS\04-Pneumonia-ClassificationProcessed/")
# fig, axis = plt.subplots(3, 3, figsize=(9, 9))
# c = 0
# for i in range(3):
#  for j in range(3):
#  patient_id = labels.patientId.iloc[c]
 
#  dcm_path = ROOT_PATH/patient_id
#  dcm_path = dcm_path.with_suffix(".dcm")
#  dcm = pydicom.read_file(dcm_path).pixel_array
 
#  label = labels["Target"].iloc[c]
 
#  axis[i][j].imshow(dcm, cmap="bone")
#  axis[i][j].set_title(label)
#  c+=1
# sums = 0
# sums_squared = 0
# for c, patient_id in enumerate(tqdm(labels.patientId)):
#  dcm_path = ROOT_PATH/patient_id
#  dcm_path = dcm_path.with_suffix(".dcm")
 
 
#  dcm = pydicom.read_file(dcm_path).pixel_array / 255 
 
#  dcm_array = cv2.resize(dcm, (224, 224)).astype(np.float16)
 
 
#  label = labels.Target.iloc[c]
 
#  train_or_val = "train" if c < 24000 else "val"
 
#  current_save_path = SAVE_PATH/train_or_val/str(label)
#  current_save_path.mkdir(parents=True, exist_ok=True)
#  np.save(current_save_path/patient_id, dcm_array)
 
#  normalizer = dcm_array.shape[0] * dcm_array.shape[1]
#  if train_or_val == "train":
#  sums += np.sum(dcm_array) / normalizer
#  sums_squared += (np.power(dcm_array, 2).sum()) / normalizer
# 0%| | 0/26684 [00:00<?, ?it/s]
# mean = sums / 24000
# std = np.sqrt(sums_squared / 24000 - (mean**2))
# print(f"Mean of Dataset: {mean}, STD: {std}")
# Mean of Dataset: 0.49039623525191567, STD: 0.2479507326197431










#Training
# import torch
# import torchvision
# from torchvision import transforms
# import torchmetrics
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import TensorBoardLogger
# from tqdm.notebook import tqdm
# import numpy as np
# import matplotlib.pyplot as plt
# def load_file(path):
#  return np.load(path).astype(np.float32)
# train_transforms = transforms.Compose([
#  transforms.ToTensor(),
# transforms.Normalize(0.49, 0.248),
#  transforms.RandomAffine(
#  degrees=(-5, 5), translate=(0, 0.05), scale=(0.9, 1.1)),
#  transforms.RandomResizedCrop((224, 224), scale=(0.35, 1))
# ])
# val_transforms = transforms.Compose([
#  transforms.ToTensor(),
#  transforms.Normalize([0.49], [0.248]),
# ])
# train_dataset = torchvision.datasets.DatasetFolder(
#  r"C:\Users\kisha\OneDrive\Desktop\Projects\MedAI\AI-IN-MEDICAL-MATERIALS\04-Pneumonia-ClassificationProcessed/train/",
#  loader=load_file, extensions="npy", transform=train_transforms)
# val_dataset = torchvision.datasets.DatasetFolder(
#  r"C:\Users\kisha\OneDrive\Desktop\Projects\MedAI\AI-IN-MEDICAL-MATERIALS\04-Pneumonia-ClassificationProcessed/val/",
#  loader=load_file, extensions="npy", transform=val_transforms)
# fig, axis = plt.subplots(2, 2, figsize=(9, 9))
# for i in range(2):
#  for j in range(2):
#  random_index = np.random.randint(0, 20000)
#  x_ray, label = train_dataset[random_index]
#  axis[i][j].imshow(x_ray[0], cmap="bone")
#  axis[i][j].set_title(f"Label:{label}")
#     batch_size = 64
# num_workers = 0
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
# print(f"There are {len(train_dataset)} train images and {len(val_dataset)} val images")
# There are 24000 train images and 2684 val images
# np.unique(train_dataset.targets, return_counts=True), np.unique(val_dataset.targets, return_counts=True)
# ((array([0, 1]), array([18593, 5407], dtype=int64)),
# (array([0, 1]), array([2079, 605], dtype=int64)))
# class PneumoniaModel(pl.LightningModule):
#  def __init__(self, weight=1):
#  super().__init__()
 
#  self.model = torchvision.models.resnet18()
 
#  self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
 
#  self.model.fc = torch.nn.Linear(in_features=512, out_features=1)
 
#  self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
#  self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))
 
#  # simple accuracy computation
#  self.train_acc = torchmetrics.Accuracy(task="binary")
#  self.val_acc = torchmetrics.Accuracy(task="binary")
#  def forward(self, data):
#  pred = self.model(data)
#  return pred
 
#  def training_step(self, batch, batch_idx):
#  x_ray, label = batch
#  label = label.float() # Convert label to float (just needed for loss computation)
#  pred = self(x_ray)[:,0] # Prediction: Make sure prediction and label have same shape
#  #while using pytorch lightening we don't need to write self.forward
#  #our prediction are of shape (batchsize,1) but our lebels are only of shape batchsize
#  #hence remove second dimension from prediction by slicing
#  loss = self.loss_fn(pred, label) # Compute the loss
 
#  # Log loss and batch accuracy
#  self.log("Train Loss", loss)
#  self.log("Step Train Acc", self.train_acc(torch.sigmoid(pred), label.int()))
#  return loss
 
 
#  def on_train_epoch_end(self):
#  # After one epoch compute the whole train_data accuracy
#  self.log("Train Acc", self.train_acc.compute())
 
 
#  def validation_step(self, batch, batch_idx):
#  # Same steps as in the training_step
#  x_ray, label = batch
#  label = label.float()
#  pred = self(x_ray)[:,0] # to make sure prediction and label have same shape
#  loss = self.loss_fn(pred, label)
 
#  # Log validation metrics
#  self.log("Val Loss", loss)
#  self.log("Step Val Acc", self.val_acc(torch.sigmoid(pred), label.int()))
#  return loss
 
#  def on_validation_epoch_end(self):
#  self.log("Val Acc", self.val_acc.compute())
 
#  def configure_optimizers(self):
#  #Caution! You always need to return a list here (just pack your optimizer into one :))
#  return [self.optimizer]
# model = PneumoniaModel()
# # Create the checkpoint callback
# checkpoint_callback = ModelCheckpoint(
#  monitor='Val Acc',
#  save_top_k=10,
#  mode='max')
# # Create the trainer
# # Change the gpus parameter to the number of available gpus on your system. Use 0 for CPU training
# #here provide whole path to your logs directory
# trainer = pl.Trainer(devices=[0], accelerator="gpu",logger=TensorBoardLogger(save_dir=r"C:\Users\kisha\OneDrive\Desktop\Projects\MedAI\AI-IN-MEDICAL-MATERIALS\04-Pneumonia-Classification/logs"), log_every_n_steps=1,
#  callbacks=checkpoint_callback,
# max_epochs=15)
# GPU available: True (cuda), used: True
# TPU available: False, using: 0 TPU cores
# IPU available: False, using: 0 IPUs
# HPU available: False, using: 0 HPUs
# trainer.fit(model, train_loader, val_loader)
# You are using a CUDA device ('NVIDIA GeForce RTX 3050 Laptop GPU') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pyto
# rch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
# LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
# | Name | Type | Params
# ------------------------------------------------
# 0 | model | ResNet | 11.2 M
# 1 | loss_fn | BCEWithLogitsLoss | 0 
# 2 | train_acc | BinaryAccuracy | 0 
# 3 | val_acc | BinaryAccuracy | 0 
# ------------------------------------------------
# 11.2 M Trainable params
# 0 Non-trainable params
# 11.2 M Total params
# 44.683 Total estimated model params size (MB)
# Sanity Checking: | | 0/? [00:00<?…
# C:\Users\kisha\anaconda3\envs\kishan\lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:441: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=15`
# in the `DataLoader` to improve performance.
# C:\Users\kisha\anaconda3\envs\kishan\lib\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:441: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=1
# 5` in the `DataLoader` to improve performance.
# Training: | | 0/? [00:00<?…
# Validation: | | 0/? [00:00<?…
# Validation: | | 0/? [00:00<?…
# Validation: | | 0/? [00:00<?…
# Validation: | | 0/? [00:00<?…
# Validation: | | 0/? [00:00<?…
# Validation: | | 0/? [00:00<?…
# Validation: | | 0/? [00:00<?…
# Validation: | | 0/? [00:00<?…
# Validation: | | 0/? [00:00<?…
# Validation: | | 0/? [00:00<?…
# Validation: | | 0/? [00:00<?…
# Validation: | | 0/? [00:00<?…
# Validation: | | 0/? [00:00<?…
# Validation: | | 0/? [00:00<?…
# Validation: | | 0/? [00:00<?…
# `Trainer.fit` stopped: `max_epochs=15` reached.
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# #We evaluate to calculate accuracy,precision,recall and confusion matrix
# # Use strict=False, otherwise we would want to match the pos_weight which is not necessary
# #write path correctly
# model = PneumoniaModel.load_from_checkpoint(r"C:\Users\kisha\OneDrive\Desktop\Projects\MedAI\AI-IN-MEDICAL-MATERIALS\04-Pneumonia-Classification\weights\weights_3.ckpt")
# model.eval()
# model.to(device);
# C:\Users\kisha\anaconda3\envs\kishan\lib\site-packages\pytorch_lightning\utilities\migration\migration.py:208: You have multiple `ModelCheckpoint` callback states in this checkpoint, but we found state keys that would end up colliding with each other after an upgrade, w
# hich means we can't differentiate which of your checkpoint callbacks needs which states. At least one of your `ModelCheckpoint` callbacks will not be able to reload the state.
# Lightning automatically upgraded your loaded checkpoint from v1.3.4 to v2.2.1. To apply the upgrade to your files permanently, run `python -m pytorch_lightning.utilities.upgrade_checkpoint C:\Users\kisha\OneDrive\Desktop\Projects\MedAI\AI-IN-MEDICAL-MATERIALS\04-Pneumon
# ia-Classification\weights\weights_3.ckpt`
# preds = []
# labels = []
# with torch.no_grad():
#  for data, label in tqdm(val_dataset):
#  data = data.to(device).float().unsqueeze(0)
#  pred = torch.sigmoid(model(data)[0].cpu())
#  preds.append(pred)
#  labels.append(label)
# preds = torch.tensor(preds)
# labels = torch.tensor(labels).int()
# 0%| | 0/2684 [00:00<?, ?it/s]
# acc = torchmetrics.Accuracy(task="binary")(preds, labels)
# precision = torchmetrics.Precision(task="binary")(preds, labels)
# recall = torchmetrics.Recall(task="binary")(preds, labels)
# cm = torchmetrics.ConfusionMatrix(num_classes=2,task="binary")(preds, labels)
# cm_threshed = torchmetrics.ConfusionMatrix(num_classes=2, threshold=0.25,task="binary")(preds, labels)
# print(f"Val Accuracy: {acc}")
# print(f"Val Precision: {precision}")
# print(f"Val Recall: {recall}")
# print(f"Confusion Matrix:\n {cm}")
# print(f"Confusion Matrix 2:\n {cm_threshed}")
# #recall is much larger,this means our model rarely misses a case of pneumonia
# #however low precision means lots of False positive
# #we can rerun the complete evaluation with the model which was not trained with weighted loss
# #this time it will fails to detect many amount of pneumonias
# Val Accuracy: 0.7652757167816162
# Val Precision: 0.48819640278816223
# Val Recall: 0.8545454740524292
# Confusion Matrix:
# tensor([[1537, 542],
#  [ 88, 517]])
# Confusion Matrix 2:
# tensor([[1135, 944],
#  [ 23, 582]])
# fig, axis = plt.subplots(2, 2, figsize=(4, 4))
# for i in range(2):
#  for j in range(2):
#  rnd_idx = np.random.randint(0, len(preds))
#  axis[i][j].imshow(val_dataset[rnd_idx][0][0], cmap="bone")
#  axis[i][j].set_title(f"Pred:{int(preds[rnd_idx] > 0.5)}, Label:{labels[rnd_idx]}")
#  axis[i][j].axis("off")
