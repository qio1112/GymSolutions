import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def load_data(data_path, batch_size=32, shuffle=False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor()
    ])

    dataset = ImageFolder(data_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, data_loader


def train_test_valid_data(train_path,
                          test_path,
                          valid_path,
                          batch_size=32):
    train_dataset, train_data_loader = load_data(train_path, batch_size, True)
    test_dataset, test_data_loader = load_data(test_path, batch_size, False)
    valid_dataset, valid_data_loader = load_data(valid_path, batch_size, False)

    return train_dataset, train_data_loader, test_dataset, test_data_loader, valid_dataset, valid_data_loader

