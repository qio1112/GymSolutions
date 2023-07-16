import torch.optim as optim
import torch.nn as nn
import torch
from tqdm.auto import tqdm
import time
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
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += len(images)

    model.train()
    return total_loss, total_correct, total_samples


def train(model, train_data_loader, test_data_loader, num_epochs=10):

    print("start training")
    lr = 0.001

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []
    for epoch in range(num_epochs):
        train_loss = 0
        train_total_correct = 0
        train_batch_count = 0
        for i, (images, labels) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
            train_batch_count += 1
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, train_prediction = torch.max(outputs, 1)
            num_train_correct = (train_prediction == labels).sum().item()
            train_total_correct += num_train_correct
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # if i % (len(train_data_loader) // 5) == 0:
            #     print(f"epoch {epoch+1}/{num_epochs}, batch {i+1}/{len(train_data_loader)}")
        time.sleep(1)
        train_loss = train_loss / train_batch_count
        train_loss_history.append(train_loss)
        train_accuracy = train_total_correct / len(train_data_loader.dataset)

        test_loss, test_correct, test_total = validate(model, test_data_loader, criterion)
        test_accuracy = test_correct/test_total
        test_loss_history.append(test_loss)
        train_acc_history.append(train_accuracy)
        test_acc_history.append(test_accuracy)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train_Loss: {train_loss:.4f}, Train_Accuracy: {train_accuracy:.4f}, Test_Loss: {test_loss:.4f}, Test_Accuracy: {test_accuracy:.4f}")
        time.sleep(1)

    return model, train_loss_history, test_loss_history, train_acc_history, test_acc_history, criterion

