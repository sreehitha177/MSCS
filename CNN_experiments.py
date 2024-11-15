transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to uniform size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


train_dataset = datasets.ImageFolder(root="/content/sample_data/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.ImageFolder(root="/content/sample_data/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


model = CNN()
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_cnn_model()

# Evaluate and save predictions
evaluate_and_save_predictions(model, test_loader)

