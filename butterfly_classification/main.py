from train import train, validate
from prep_data import train_test_valid_data
from utils.utils import draw_loss_accuracy_history, save_model
from model import CNNModel, CNNModel2
import os

if __name__ == "__main__":
    full_data = True
    batch_size = 256
    num_epochs = 50
    num_classes = 100 if full_data else 3

    # put data under ./data/archive/train or ./data/reduced/train
    path = os.getcwd()
    root_path = os.path.join(path, "data", "archive" if full_data else "reduced")
    train_path = os.path.join(root_path, "train")
    test_path = os.path.join(root_path, "test")
    valid_path = os.path.join(root_path, "valid")

    model = CNNModel2(num_classes)
    # model = CNNModel(num_classes)

    _, train_data_loader, _, test_data_loader, _, valid_data_loader = train_test_valid_data(train_path,
                                                                                            test_path,
                                                                                            valid_path,
                                                                                            batch_size=batch_size)
    model, train_loss_history, test_loss_history, train_acc, test_acc, criterion = train(model,
                                                                                         train_data_loader,
                                                                                         test_data_loader,
                                                                                         num_epochs=num_epochs)

    valid_loss, valid_correct, valid_total = validate(model, valid_data_loader, criterion)
    correct_rate = valid_correct / valid_total
    valid_result = f"Validating_Loss: {valid_loss:.4f}, Correct Rate: {correct_rate:.4f} ({valid_correct} correct out of {valid_total})"
    print(valid_result)

    history = {
        'train_loss': train_loss_history,
        'train_accuracy': train_acc,
        'test_loss': test_loss_history,
        'test_accuracy': test_acc,
        'validation_result': [valid_result]
    }

    save_path = save_model(path, model, 'CNNModel', None, history, False)
    draw_loss_accuracy_history(train_loss_history, test_loss_history, train_acc, test_acc, save_path)
