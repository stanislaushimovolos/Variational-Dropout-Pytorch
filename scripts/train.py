import json
import argparse
from torch.optim import Adam
from torch import nn as nn
from var_drop import make_mnist_data_loaders, train, inference_step, LeNet, ELBOLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--checkpoints_path', type=str, help='path to save model parameters and train artifacts')

    args = parser.parse_args()
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    checkpoints_path = args.checkpoints_path

    model = LeNet()
    model.cuda()
    optimizer = Adam(model.parameters(), lr=5e-4)
    train_loader, test_loader = make_mnist_data_loaders(batch_size)

    test_loss_history = []

    def callback():
        criterion = nn.CrossEntropyLoss()
        n_total = 0
        n_correct = 0

        test_losses = []
        for inputs, targets in test_loader:
            preds = inference_step(inputs, model, use_amp=False).cpu()
            loss = criterion(preds, targets)
            test_losses.append(loss.item())

            _, predicted = preds.max(1)
            n_total += targets.shape[0]
            n_correct += (predicted == targets).sum().item()

        test_loss_history.append(test_losses)
        return n_correct / n_total

    train_loss_history = train(
        model=model,
        scaler=None,
        optimizer=optimizer,
        n_epochs=n_epochs,
        validation_callback=callback,
        train_data_loader=train_loader,
        criterion=nn.CrossEntropyLoss(),
        checkpoints_path=checkpoints_path,
    )

    with open(checkpoints_path / 'train_losses.json') as f:
        json.dump(train_loss_history, f)

    with open(checkpoints_path / 'test_losses.json') as f:
        json.dump(test_loss_history, f)


if __name__ == '__main__':
    main()
