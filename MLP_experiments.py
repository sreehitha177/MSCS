# import os
# import pandas as pd
#
# test_files = os.listdir("../data/test/unlabeled")
# test_files.sort()
# n = len(test_files)
# labels =["red_blood_cell"]*n
#
# df = pd.DataFrame({"Input": test_files, "Class":labels})
# df.to_csv("mlp_predictions.csv",index=False)


import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# import MLP, train_mlp_model, evaluate_and_save_predictions


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
train_dataset = datasets.ImageFolder(root='/content/sample_data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = datasets.ImageFolder(root='/content/sample_data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Set model parameters
input_size = 128 * 128 * 3
hidden_size1 = 256
hidden_size2 = 128
hidden_size3 = 64
num_classes = 6

# Initialize model, criterion, and optimizer
model = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Train the model
train_mlp_model(model, train_loader, criterion, optimizer, num_epochs=10)

# Evaluate and save predictions
evaluate_and_save_predictions(model, test_loader, output_file="mlp_predictions.csv")
