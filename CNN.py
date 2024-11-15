import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd

class_labels = ["gametocyte", "leukocyte", "red_blood_cell", "ring", "schizont", "trophozoite"]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 6)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

def train_cnn_model():
  epochs = 10
  for epoch in range(epochs):
      model.train()  # Set the model to training mode
      running_loss = 0.0
      correct = 0
      total = 0
      for inputs, labels in train_loader:
          optimizer.zero_grad()  # Zero the gradients

          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          running_loss += loss.item()

          _, predicted = torch.max(outputs, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()


      accuracy = correct / total * 100
      print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy}%")


def evaluate_and_save_predictions(model, test_loader):
  predictions = []

  with torch.no_grad():
      for inputs, _ in test_loader:
          outputs = model(inputs)
          _, predicted = torch.max(outputs, 1)
          # predictions.extend(predicted.numpy())
          predicted_labels = [class_labels[idx] for idx in predicted]
          predictions.extend(predicted_labels)


  # Prepare predictions for CSV file
  test_files = os.listdir("/content/sample_data/test/unlabeled")
  test_files.sort()

  print(f"Length of test_files: {len(test_files)}")
  print(f"Length of predictions: {len(predictions)}")

  min_length = min(len(test_files), len(predictions))
  test_files = test_files[:min_length]
  predictions = predictions[:min_length]

  df = pd.DataFrame({"Input": test_files, "Class": predictions})
  df.to_csv("cnn_predictions.csv", index=False)
  print("Predictions saved to 'cnn_predictions.csv'.")
