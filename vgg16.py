# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import resize, rotate

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import torch
from torch import optim
from torch.nn import Linear, CrossEntropyLoss, Sequential
from torch.autograd import Variable

from torchvision import models

# get files from drive
import os

dir = "./"
train_norm_dir = dir+"frames/train/norm/"
train_weap_dir = dir+"frames/train/weap/"
test_norm_dir = dir+"frames/test/norm/"
test_weap_dir = dir+"frames/test/weap/"

def get_images(path):
  images = []
  print("reading from "+path)
  num_files = len(os.listdir(path))
  print(str(num_files) + " files to read")
  num_files_read = 0
  next_percentage = 1
  for filename in os.listdir(path):
    if 10*num_files_read / num_files > next_percentage:
      print(str(next_percentage * 10) + "% finished")
      next_percentage+=1

    f = os.path.join(path, filename)
    if os.path.join(f):
      img = imread(f)
      img = img/255
      img = resize(img, output_shape=(224, 224, 3), mode='constant', anti_aliasing=True)

      img = img.astype('float32')
      images.append(img)
    num_files_read += 1

  return images

num_train_norm = len(os.listdir(train_norm_dir))
num_train_weap = len(os.listdir(train_weap_dir))
num_test_norm = len(os.listdir(test_norm_dir))
num_test_weap = len(os.listdir(test_weap_dir))

train_x_raw = get_images(train_norm_dir) + get_images(train_weap_dir)
test_x_raw = get_images(test_norm_dir) + get_images(test_weap_dir)

train_y_raw = []
for i in range(num_train_norm):
  train_y_raw.append(0)
for i in range(num_train_weap):
  train_y_raw.append(1)

test_y_raw = []
for i in range(num_test_norm):
  test_y_raw.append(0)
for i in range(num_test_weap):
  test_y_raw.append(1)

train_x = np.array(train_x_raw)
train_y = np.array(train_y_raw)

test_x = np.array(test_x_raw)
test_y = np.array(test_y_raw)

# converting training images into torch format
train_x = train_x.reshape((num_train_norm+num_train_weap), 3, 224, 224)
train_x = torch.from_numpy(train_x)
train_y = train_y.astype(int)
train_y = torch.from_numpy(train_y)

# converting test images into torch format
test_x = test_x.reshape(num_test_norm+num_test_weap, 3, 224, 224)
test_x = torch.from_numpy(test_x)
test_y = test_y.astype(int)
test_y = torch.from_numpy(test_y)

# loading the pretrained model
model = models.vgg16_bn(pretrained=True)

# checking if GPU is available
if torch.cuda.is_available():
    model.cuda()

# Freeze model weights of the VGG-16 model.
for param in model.parameters():
   param.requires_grad = False

# Add a Linear layer to the classifier
model.classifier[6] = Sequential(
    Linear(4096, 2).cuda())

#Train the model by updating the weights of the last layer
for param in model.classifier[6].parameters():
    param.requires_grad = True

# batch_size
batch_size = 64

# extracting features for train data
data_x = []
label_x = []

inputs, labels = train_x, train_y
inputs = inputs.cuda()

for i in range(int(train_x.shape[0]/batch_size)+1):
    input_data = inputs[i*batch_size:(i+1)*batch_size]
    label_data = labels[i*batch_size:(i+1)*batch_size]
    input_data, label_data = Variable(
        input_data.cuda()), Variable(label_data.cuda())
    x = model.features(input_data)
    data_x.extend(x.data.cpu().numpy())
    label_x.extend(label_data.data.cpu().numpy())

# extracting features for test data
data_z = []
label_z = []

inputs, labels = test_x, test_y

for i in range(int(test_x.shape[0]/batch_size)+1):
    input_data = inputs[i*batch_size:(i+1)*batch_size]
    label_data = labels[i*batch_size:(i+1)*batch_size]
    input_data, label_data = Variable(
        input_data.cuda()), Variable(label_data.cuda())
    x = model.features(input_data)
    data_z.extend(x.data.cpu().numpy())
    label_z.extend(label_data.data.cpu().numpy())

# converting training images and its labels into torch format
x_train = torch.from_numpy(np.array(data_x))
x_train = x_train.view(x_train.size(0), -1)
y_train = torch.from_numpy(np.array(label_x))

# converting test images and its labels into torch format
x_test  = torch.from_numpy(np.array(data_z))
x_test = x_test.view(x_test.size(0), -1)
y_test  = torch.from_numpy(np.array(label_z))

# specify loss function (categorical cross-entropy)
criterion = CrossEntropyLoss()

# specify optimizer (Adam) and learning rate
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.0005)

# batch size of the model
batch_size = 64

# number of epochs to train the model
n_epochs = 20

for epoch in range(1, n_epochs+1):

    train_loss = 0.0
    permutation = torch.randperm(x_train.size()[0])
    training_loss = []
    for i in range(0,x_train.size()[0], batch_size):

        indices = permutation[i:i+batch_size]
        batch_x, batch_y = x_train[indices], y_train[indices]

        if torch.cuda.is_available():
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        optimizer.zero_grad()
        outputs = model.classifier(batch_x.cuda())
        loss = criterion(outputs,batch_y)

        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    training_loss = np.average(training_loss)
    print('epoch: \t', epoch, '\t Training loss: \t', training_loss)

# prediction for training set
prediction = []
target = []
permutation = torch.randperm(x_train.size()[0])
for i in range(0, x_train.size()[0], batch_size):
    indices = permutation[i:i+batch_size]
    batch_x, batch_y = x_train[indices], y_train[indices]

    if torch.cuda.is_available():
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

    with torch.no_grad():
        output = model.classifier(batch_x.cuda())

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    prediction.append(predictions)
    target.append(batch_y)

# Training accuracy
accuracy = []
for i in range(len(prediction)):
    accuracy.append(accuracy_score(target[i].cpu(), prediction[i]))

print('Training accuracy: \t', np.average(accuracy))

# prediction for Test set
prediction_test = []
target_test = []

permutation = torch.randperm(x_test.size()[0])
for i in range(0, x_test.size()[0], batch_size):
    indices = permutation[i:i+batch_size]
    batch_x, batch_y = x_test[indices], y_test[indices]

    if torch.cuda.is_available():
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

    with torch.no_grad():
        output = model.classifier(batch_x.cuda())

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())

    predictions = np.argmax(prob, axis=1)
    prediction_test.append(predictions)
    target_test.append(batch_y)

# Test accuracy
accuracy_test = []
for i in range(len(prediction_test)):
    accuracy_test.append(accuracy_score(target_test[i].cpu(), prediction_test[i]))

print('Test accuracy: \t', np.average(accuracy_test))
torch.save(model, 'VGG16-model.pth')

# confusion matrix
confusion_matrices = []
for i in range(len(prediction_test)):
    confusion_matrices.append(confusion_matrix(target_test[i].cpu(), prediction_test[i]))
confusion_matrices = np.array(confusion_matrices)
confmat = confusion_matrices.sum(axis=0)
print('Confusion Matrix: \t', confmat)

# precision
precision = confmat[1][1] / (confmat[1][1] + confmat[0][1])
print('Precision: \t', precision)

# recall
recall = confmat[1][1] / (confmat[1][1] + confmat[1][0])
print('Recall: \t', recall)

# f1 score
f1 = 2 * (precision * recall) / (precision + recall)
print('F1 Score: \t', f1)
