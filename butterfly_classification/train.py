import torch.optim as optim
import torch.nn as nn
import torch
from prep_data import train_test_valid_data
from model import CNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def validate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images.to(device)
            labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    model.train()
    return total_loss, total_correct, total_samples


def train(train_data_loader, test_data_loader, num_classes=100, num_epochs=10):

    print("start training")
    lr = 0.001

    model = CNNModel(num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_history = []
    test_loss_history = []
    for epoch in range(num_epochs):
        train_loss = 0
        for i, (images, labels) in enumerate(train_data_loader):
            print(f"epoch {epoch}/{num_epochs}, batch {i}/{len(train_data_loader)}")
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss_history.append(train_loss)
        test_loss = validate(model, test_data_loader, criterion)
        test_loss_history.append(test_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train_Loss: {train_loss:.4f}, Test_Loss: {test_loss:.4f}")

    return model, train_loss_history, test_loss_history, criterion

