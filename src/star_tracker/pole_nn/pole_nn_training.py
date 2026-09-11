"""
Train model
"""

import torch
import optuna
import numpy as np
import pandas as pd
import h5py
import argparse
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from .pole_nn import pole_nn, H5Data

ROOT = Path(__file__).resolve().parent.parent.parent.parent

def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return number


def positive_float(value):
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return number


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="cuda",
        help="Device used to run the model, such as cuda, cpu, or mps "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "-b", "--bins",
        type=positive_int,
        default=25,
        help="Number of distance bins used in data generation (default: %(default)s)",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Randomization seed (default: %(default)s)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    hypertune_parser = subparsers.add_parser(
        "hypertune",
        help="Hypertune the model parameters",
    )
    hypertune_parser.add_argument(
        "--hidden-size-1",
        nargs=3,
        type=positive_int,
        default=[64, 128, 16],
        metavar=("MIN", "MAX", "STEP"),
        help="Search range for hidden layer 1 (default: 64 128 16)",
    )
    hypertune_parser.add_argument(
        "--hidden-size-2",
        nargs=3,
        type=positive_int,
        default=[64, 128, 16],
        metavar=("MIN", "MAX", "STEP"),
        help="Search range for hidden layer 2 (default: 64 128 16)",
    )
    hypertune_parser.add_argument(
        "--lr",
        nargs=2,
        type=positive_float,
        default=[1e-4, 1e-1],
        metavar=("MIN", "MAX"),
        help="Learning-rate search range (default: 1e-4 1e-1)",
    )
    hypertune_parser.add_argument(
        "--n-trials",
        type=positive_int,
        default=50,
        help="Number of hypertuning trials (default: %(default)s)",
    )
    hypertune_parser.add_argument(
        "--save-trials",
        action="store_true",
        help="Save Optuna trials to a CSV file",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train a model and save its weights",
    )
    train_parser.add_argument(
        "--hidden-size-1",
        type=positive_int,
        default=128,
        help="Size of hidden layer 1 (default: %(default)s)",
    )
    train_parser.add_argument(
        "--hidden-size-2",
        type=positive_int,
        default=128,
        help="Size of hidden layer 2 (default: %(default)s)",
    )
    train_parser.add_argument(
        "-e", "--epochs",
        type=positive_int,
        default=5,
        help="Number of training epochs (default: %(default)s)",
    )
    train_parser.add_argument(
        "--lr",
        type=positive_float,
        default=1e-3,
        help="Learning rate used for training (default: %(default)s)",
    )
    train_parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save model weights to a .pth file",
    )

    test_parser = subparsers.add_parser(
        "test",
        help="Test the model against a test set",
    )
    test_parser.add_argument(
        "--hidden-size-1",
        type=positive_int,
        default=128,
        help="Size of hidden layer 1 used by the saved model (default: %(default)s)",
    )
    test_parser.add_argument(
        "--hidden-size-2",
        type=positive_int,
        default=128,
        help="Size of hidden layer 2 used by the saved model (default: %(default)s)",
    )
    test_parser.add_argument(
        "--file-name",
        default="model_weights.pth",
        help="Name of the full .pth file used to load in the weights (default: %(default)s)",
    )

    args = parser.parse_args()

    if args.command == "hypertune":
        for option in ("hidden_size_1", "hidden_size_2"):
            minimum, maximum, step = getattr(args, option)
            if minimum > maximum:
                parser.error(f"--{option.replace('_', '-')} MIN cannot exceed MAX")
            if step > maximum - minimum and minimum != maximum:
                parser.error(
                    f"--{option.replace('_', '-')} STEP is larger than its range"
                )

        lr_min, lr_max = args.lr
        if lr_min > lr_max:
            parser.error("--lr MIN cannot exceed MAX")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(
            f"CUDA device '{args.device}' was requested, but CUDA is unavailable. "
            "Use `--device cpu` instead."
        )

    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error(
            "MPS was requested, but MPS is unavailable. "
            "Use `--device cpu` instead."
        )
    return args

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    with torch.inference_mode():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            total_loss += loss_fn(model(x), y).item()

    return total_loss / len(dataloader)

def objective(trial, train_dataloader, val_dataloader, device, args):
    hidden_size_one = trial.suggest_int("hidden_size_one", args.hidden_size_1[0], args.hidden_size_1[1], step=args.hidden_size_1[2])
    hidden_size_two = trial.suggest_int("hidden_size_two", args.hidden_size_2[0], args.hidden_size_2[1], step=args.hidden_size_2[2])
    learning_rate = trial.suggest_float("lr", args.lr[0], args.lr[1], log=True)

    model = pole_nn(
        n_bins=args.bins,
        n_classes=9029,
        hidden_one=hidden_size_one,
        hidden_two=hidden_size_two
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()

        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

        val_loss = evaluate(model, val_dataloader, loss_fn, device)
        trial.report(val_loss, step=epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss


def hypertune(args, train_dataloader, val_dataloader, device):
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    study.optimize(lambda trial: objective(trial, train_dataloader, val_dataloader, device, args), n_trials=args.n_trials)

    print(study.trials_dataframe())
    print("\n")
    print("Best loss:", study.best_value)
    print("Best parameters:", study.best_params)

    if args.save_trials:
        name = input("Enter trials csv name (with.csv)")
        data_path = ROOT / "data" / name
        study.trials_dataframe().to_csv(data_path, index=False)


def train(args, train_dataloader, test_dataloader, device):
    model = pole_nn(args.bins, 9029, args.hidden_size_1, args.hidden_size_2)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr)

    BATCH_SIZE = 32
    print("Beginning training!")
    for epoch in range(args.epochs):
        train_loss, train_accuracy = 0, 0
        model.train()
        for batch, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            train_pred = torch.argmax(y_pred, dim=1)
            train_accuracy += (train_pred == y).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)
        train_accuracy /= len(train_dataloader) * BATCH_SIZE

        test_loss, test_accuracy = 0, 0
        model.eval()
        with torch.inference_mode():
            for x,y in (test_dataloader):
                x, y = x.to(device), y.to(device)

                y_pred = model(x)
                test_loss += loss_fn(y_pred, y).item()
                test_pred = torch.argmax(y_pred, dim=1)
                test_accuracy += (test_pred == y).sum().item()

            test_loss /= len(test_dataloader)
            test_accuracy /= len(test_dataloader) * BATCH_SIZE
        print(f"Epoch: {epoch}")
        print(f"Train loss: {train_loss:.5f} | Train Accuracy: {train_accuracy:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_accuracy:.2f}\n")

    if args.save_model:
        name = input("Input name for weights file (with .pth)")
        data_path = ROOT / "data" / name
        torch.save(model.state_dict(), data_path)


def test(args, test_dataloader, device):
    model = pole_nn(args.bins, 9029, args.hidden_size_1, args.hidden_size_2)
    model.to(device)
    model_path = ROOT / "data" / args.file_name
    model.load_state_dict(torch.load(model_path, weights_only=True))
    loss_fn = nn.CrossEntropyLoss()

    test_loss, test_accuracy = 0, 0
    model.eval()
    BATCH_SIZE = 32
    print("Starting Testing!")
    with torch.inference_mode():
        for x,y in (test_dataloader):
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            test_loss += loss_fn(y_pred, y).item()
            test_pred = torch.argmax(y_pred, dim=1)
            test_accuracy += (test_pred == y).sum().item()

        test_loss /= len(test_dataloader)
        test_accuracy /= len(test_dataloader) * BATCH_SIZE

        print(f"Test loss: {test_loss:.5f}, Test acc: {test_accuracy:.2f}\n")


def main():
    data_path = ROOT / "data" / "data.hdf5"
    data_path = Path(data_path)

    if not data_path.exists():
        FileNotFoundError(f"{data_path} is missing. Please generate the data with pole_nn_data_gen.py")
    
    args = parse_arguments()

    print("Loading hdf5 data...")
    data = H5Data(data_path)
    print("Done!")
    labels = [int(float(group)) for group, _ in data.index_map]

    #could consider adding user input for dataset sizes and batch sizes
    train_idx, temp_idx = train_test_split(
        range(len(data.index_map)),
        train_size=0.8,
        stratify=labels,
        random_state=args.seed
    )

    temp_labels = [labels[i] for i in temp_idx]

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=temp_labels,
        random_state=args.seed,
    )

    train_sub = Subset(data, train_idx)
    val_sub = Subset(data, val_idx)
    test_sub = Subset(data, test_idx)

    BATCH_SIZE = 32
    print(f"Loading Dataloaders... BATCHSIZE = {BATCH_SIZE}")
    train_dataloader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_sub, batch_size=BATCH_SIZE)
    print(f"Length of train dataloader: {len(train_dataloader)}")
    print(f"Length of val dataloader: {len(val_dataloader)}")
    print(f"Length of test dataloader: {len(test_dataloader)} \n")
        
    device = torch.device(args.device)
    print(f"Using device: {device} \n")

    if args.command == "hypertune":
        hypertune(args, train_dataloader, val_dataloader, device)
    elif args.command == "train":
        train(args, train_dataloader, test_dataloader, device)
    else:
        test(args, test_dataloader, device)

if __name__ == "__main__":
    main()