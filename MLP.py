import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

# Define class labels
class_labels = ["gametocyte", "leukocyte", "red_blood_cell", "ring", "schizont", "trophozoite"]

# Model Construction
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.softmax(x)
        return x

# Training function
def train_mlp_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        accuracy = correct_predictions / total_predictions * 100
        error_rate = np.mean(total_predictions != labels)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy}%")
#         print(f"Error rate : {error_rate}")
    # train_error_rate = np.mean(train_pred != Ytr)
    print(f"Training error rate: {error_rate}")


# Evaluation function
def evaluate_and_save_predictions(model, test_loader, output_file="mlp_predictions.csv"):
    predictions = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            # Convert predictions to class labels
            predicted_labels = [class_labels[idx] for idx in predicted]
            predictions.extend(predicted_labels)

    # Prepare predictions for CSV file
    test_files = os.listdir("/content/sample_data/test/unlabeled")
    # test_files = os.listdir("../data/test/unlabeled")
    test_files.sort()

    # print(f"Length of test_files: {len(test_files)}")
    # print(f"Length of predictions: {len(predictions)}")

    min_length = min(len(test_files), len(predictions))
    test_files = test_files[:min_length]
    predictions = predictions[:min_length]

    df = pd.DataFrame({"Input": test_files, "Class": predictions})
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to '{output_file}'.")
