import json
import argparse
from pathlib import Path
from torch.optim import Adam
from torch import nn as nn
from var_drop import VanillaLeNet, ARDLeNet, make_mnist_data_loaders, train, inference_step, ELBOLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_ard', action="store_true")
    parser.add_argument('--n_epochs', type=int, default=75)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--checkpoints_path', type=str, help='path to save model parameters and train artifacts')

    args = parser.parse_args()
    use_ard = args.use_ard
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    checkpoints_path = Path(args.checkpoints_path)

    model = ARDLeNet() if use_ard else VanillaLeNet()
    model.cuda()
    optimizer = Adam(model.parameters(), lr=1e-4)
    train_loader, test_loader = make_mnist_data_loaders(batch_size)
    criterion = ELBOLoss(model, nn.CrossEntropyLoss())

    test_loss_history = []
    validation_scores = []

    def callback():
        n_total = 0
        n_correct = 0

        test_losses = []
        for inputs, targets in test_loader:
            preds = inference_step(inputs, model, use_amp=False).cpu()
            loss, detached_loss = criterion(preds, targets)
            test_losses.append(detached_loss)

            _, predicted = preds.max(1)
            n_total += targets.shape[0]
            n_correct += (predicted == targets).sum().item()

        score = n_correct / n_total
        test_loss_history.append(test_losses)
        validation_scores.append(score)
        return score

    train_loss_history = train(
        model=model,
        scaler=None,
        n_epochs=n_epochs,
        criterion=criterion,
        optimizer=optimizer,
        validation_callback=callback,
        train_data_loader=train_loader,
        checkpoints_path=checkpoints_path,
    )

    with open(checkpoints_path / 'train_losses.json', 'w') as f:
        json.dump(train_loss_history, f)

    with open(checkpoints_path / 'test_losses.json', 'w') as f:
        json.dump(test_loss_history, f)

    with open(checkpoints_path / 'test_scores.json', 'w') as f:
        json.dump(validation_scores, f)


if __name__ == '__main__':
    main()
