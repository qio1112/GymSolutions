from train import train, validate
from prep_data import train_test_valid_data
from utils.utils import draw_train_test_loss_history
import os

if __name__ == "__main__":

    path = os.getcwd()
    train_path = os.path.join(path, "data", "archive", "train")
    test_path = os.path.join(path, "data", "archive", "test")
    valid_path = os.path.join(path, "data", "archive", "valid")

    _, train_data_loader, _, test_data_loader, _, valid_data_loader = train_test_valid_data(train_path,
                                                                                            test_path,
                                                                                            valid_path,
                                                                                            batch_size=128)
    model, train_loss_history, test_loss_history, criterion = train(train_data_loader,
                                                                    test_data_loader,
                                                                    num_classes=100,
                                                                    num_epochs=1)
    draw_train_test_loss_history(train_loss_history, test_loss_history)

    valid_loss, valid_correct, valid_total = validate(model, valid_data_loader, criterion)
    correct_rate = valid_correct / valid_total
    print(f"Validating_Loss: {valid_loss:.4f}, "
          f"Correct Rate: {correct_rate:.4f} ({valid_correct} correct out of {valid_total})")

