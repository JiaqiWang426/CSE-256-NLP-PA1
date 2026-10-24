# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import (
    glove_read_X_y, glove_TwoLayerMLP,
    embeddings_read_X_y, embeddings_TwoLayerMLP, training_and_evaluation
)
from bpe import train_bpe, bpe_read_X_y

# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy


def save_dan_history(history, model_name, output_dir):
    """Save measured train/dev metrics and their loss/accuracy curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["epoch", "train_loss", "dev_loss", "train_acc", "dev_acc"]
    csv_path = output_dir / f"{model_name}_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(fields)
        writer.writerows(zip(*(history[field] for field in fields)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train")
    axes[0].plot(history["epoch"], history["dev_loss"], label="Dev")
    axes[0].set_title("Cross-entropy loss")
    axes[0].set_ylabel("Loss")
    axes[1].plot(history["epoch"], [100 * a for a in history["train_acc"]], label="Train")
    axes[1].plot(history["epoch"], [100 * a for a in history["dev_acc"]], label="Dev")
    axes[1].set_title("Accuracy")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(model_name)
    fig.tight_layout()
    plot_path = output_dir / f"{model_name}_curves.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Metrics saved to {csv_path}")
    print(f"Curves saved to {plot_path}")


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs for DAN models')
    parser.add_argument('--output-dir', default='.', help='Directory for DAN plots and metric CSV files')
    parser.add_argument('--vocab-size', type=int, default=1000, help='Target BPE vocabulary size, including special tokens')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='DAN batch size; defaults to 64 for SUBWORDDAN and full batch for word models')
    parser.add_argument('--seed', type=int, default=None, help='Optional random seed for reproducible experiments')

    # Parse the command-line arguments
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.model == "BOW":
        start_time = time.time()
        train_data = SentimentDatasetBOW("data/train.txt")
        dev_data = SentimentDatasetBOW("data/dev.txt", vectorizer=train_data.vectorizer, train=False)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)
        print(f"Data loaded in : {time.time() - start_time} seconds")

        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model in ("Glove_DAN", "Embeddings_DAN", "SUBWORDDAN"):
        if args.epochs < 1:
            parser.error("--epochs must be positive")
        start_time = time.time()
        batch_size = args.batch_size
        output_name = args.model
        summary = {"model": args.model, "epochs": args.epochs, "seed": args.seed,
                   "optimizer": "Adam", "learning_rate": 0.001,
                   "embedding_dim": 300, "hidden_dim": 100, "dropout": 0.2}
        if args.model == "SUBWORDDAN":
            if args.vocab_size < 1:
                parser.error("--vocab-size must be positive")
            print(f"Training BPE on data/train.txt (target vocabulary {args.vocab_size})...", flush=True)
            bpe_start = time.perf_counter()
            try:
                merges, word_indexer = train_bpe('data/train.txt', args.vocab_size)
            except ValueError as error:
                parser.error(str(error))
            summary["bpe_training_seconds"] = time.perf_counter() - bpe_start
            summary["target_vocab_size"] = args.vocab_size
            summary["actual_vocab_size"] = len(word_indexer)
            summary["merge_count"] = len(merges)
            print(f"BPE learned {len(merges)} merges; actual vocabulary {len(word_indexer)}; "
                  f"time {summary['bpe_training_seconds']:.2f} seconds", flush=True)
            # The same rules, vocabulary and cache serve both datasets. Dev never fits BPE.
            cache = {}
            X_train, y_train = bpe_read_X_y('data/train.txt', merges, word_indexer, cache)
            X_dev, y_dev = bpe_read_X_y('data/dev.txt', merges, word_indexer, cache)
            model = embeddings_TwoLayerMLP(vocab_size=len(word_indexer))
            batch_size = 64 if batch_size is None else batch_size
            output_name = f"SUBWORDDAN_vocab{args.vocab_size}"
            for split, X in (("train", X_train), ("dev", X_dev)):
                lengths = (X != 0).sum(dim=1)
                summary[f"{split}_mean_subwords"] = lengths.float().mean().item()
                summary[f"{split}_max_subwords"] = lengths.max().item()
                print(f"{split}: mean subwords {lengths.float().mean().item():.2f}, "
                      f"max subwords {lengths.max().item()}", flush=True)
        else:
            Embeddings = read_word_embeddings('data/glove.6B.300d-relativized.txt')
            if args.model == "Glove_DAN":
                dim = Embeddings.get_embedding_length()
                X_train, y_train = glove_read_X_y(Embeddings, dim, 'data/train.txt')
                X_dev, y_dev = glove_read_X_y(Embeddings, dim, 'data/dev.txt')
                model = glove_TwoLayerMLP(dim=dim)
            else:
                vocab_size = len(Embeddings.word_indexer)
                X_train, y_train = embeddings_read_X_y(Embeddings, 'data/train.txt')
                X_dev, y_dev = embeddings_read_X_y(Embeddings, 'data/dev.txt')
                model = embeddings_TwoLayerMLP(vocab_size=vocab_size)
        summary["data_preparation_seconds"] = time.time() - start_time
        summary["batch_size"] = len(y_train) if batch_size is None else batch_size
        print(f"Data prepared in {summary['data_preparation_seconds']:.2f} seconds; "
              f"batch size {summary['batch_size']}", flush=True)

        training_start = time.perf_counter()
        history = training_and_evaluation(
            X_train=X_train, y_train=y_train, X_dev=X_dev, y_dev=y_dev,
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
            criterion=nn.CrossEntropyLoss(), epochs=args.epochs, batch_size=batch_size
        )
        summary["training_and_evaluation_seconds"] = time.perf_counter() - training_start
        save_dan_history(history, output_name, args.output_dir)
        best = max(range(len(history["epoch"])), key=lambda i: history["dev_acc"][i])
        summary["best_recorded_dev_accuracy"] = history["dev_acc"][best]
        summary["best_recorded_epoch"] = history["epoch"][best]
        summary["final_dev_accuracy"] = history["dev_acc"][-1]
        summary_path = Path(args.output_dir) / f"{output_name}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"Best recorded dev accuracy: {history['dev_acc'][best]:.4f} "
            f"at epoch {history['epoch'][best]}"
        )
        print(f"Training and evaluation time: {summary['training_and_evaluation_seconds']:.2f} seconds")
        print(f"Experiment settings and timing saved to {summary_path}")
    else:
        parser.error(f"Unknown model: {args.model}")


if __name__ == "__main__":
    main()
#python main.py --model Glove_DAN --epochs 200
#python main.py --model Embeddings_DAN --epochs 200
#python main.py --model SUBWORDDAN --vocab-size 1000 --epochs 30 --seed 42

r'''
PS E:\OneDrive\Desktop\UCSD\CSE256 NLP\CSE256_PA1_FA26\CSE256_PA1_FA26> python main.py --model Glove_DAN --epochs 200
Read in 14923 vectors of size 300
Data loaded in : 1.5045292377471924 seconds
Epoch 5: train loss 0.6711, dev loss 0.6693, train accuracy 0.6159, dev accuracy 0.6124
Epoch 10: train loss 0.6459, dev loss 0.6410, train accuracy 0.6842, dev accuracy 0.7087
Epoch 15: train loss 0.6185, dev loss 0.6113, train accuracy 0.7014, dev accuracy 0.7248
Epoch 20: train loss 0.5901, dev loss 0.5811, train accuracy 0.7223, dev accuracy 0.7489
Epoch 25: train loss 0.5615, dev loss 0.5529, train accuracy 0.7406, dev accuracy 0.7615
Epoch 30: train loss 0.5340, dev loss 0.5276, train accuracy 0.7526, dev accuracy 0.7569
Epoch 35: train loss 0.5098, dev loss 0.5066, train accuracy 0.7686, dev accuracy 0.7672
Epoch 40: train loss 0.4898, dev loss 0.4906, train accuracy 0.7770, dev accuracy 0.7706
Epoch 45: train loss 0.4740, dev loss 0.4785, train accuracy 0.7822, dev accuracy 0.7718
Epoch 50: train loss 0.4618, dev loss 0.4693, train accuracy 0.7873, dev accuracy 0.7729
Epoch 55: train loss 0.4522, dev loss 0.4630, train accuracy 0.7902, dev accuracy 0.7775
Epoch 60: train loss 0.4444, dev loss 0.4589, train accuracy 0.7942, dev accuracy 0.7821
Epoch 65: train loss 0.4380, dev loss 0.4562, train accuracy 0.7957, dev accuracy 0.7878
Epoch 70: train loss 0.4326, dev loss 0.4545, train accuracy 0.7994, dev accuracy 0.7878
Epoch 75: train loss 0.4279, dev loss 0.4530, train accuracy 0.8016, dev accuracy 0.7913
Epoch 80: train loss 0.4240, dev loss 0.4521, train accuracy 0.8007, dev accuracy 0.7867
Epoch 85: train loss 0.4205, dev loss 0.4512, train accuracy 0.8030, dev accuracy 0.7821
Epoch 90: train loss 0.4174, dev loss 0.4504, train accuracy 0.8052, dev accuracy 0.7821
Epoch 95: train loss 0.4145, dev loss 0.4506, train accuracy 0.8074, dev accuracy 0.7810
Epoch 100: train loss 0.4119, dev loss 0.4506, train accuracy 0.8082, dev accuracy 0.7833
Epoch 105: train loss 0.4095, dev loss 0.4506, train accuracy 0.8087, dev accuracy 0.7821
Epoch 110: train loss 0.4072, dev loss 0.4507, train accuracy 0.8082, dev accuracy 0.7821
Epoch 115: train loss 0.4050, dev loss 0.4510, train accuracy 0.8091, dev accuracy 0.7878
Epoch 120: train loss 0.4028, dev loss 0.4509, train accuracy 0.8097, dev accuracy 0.7844
Epoch 125: train loss 0.4008, dev loss 0.4508, train accuracy 0.8124, dev accuracy 0.7867
Epoch 130: train loss 0.3989, dev loss 0.4510, train accuracy 0.8136, dev accuracy 0.7878
Epoch 135: train loss 0.3970, dev loss 0.4510, train accuracy 0.8145, dev accuracy 0.7878
Epoch 140: train loss 0.3951, dev loss 0.4519, train accuracy 0.8156, dev accuracy 0.7844
Epoch 145: train loss 0.3933, dev loss 0.4527, train accuracy 0.8169, dev accuracy 0.7867
Epoch 150: train loss 0.3915, dev loss 0.4527, train accuracy 0.8173, dev accuracy 0.7890
Epoch 155: train loss 0.3897, dev loss 0.4532, train accuracy 0.8186, dev accuracy 0.7856
Epoch 160: train loss 0.3880, dev loss 0.4535, train accuracy 0.8204, dev accuracy 0.7856
Epoch 165: train loss 0.3862, dev loss 0.4539, train accuracy 0.8212, dev accuracy 0.7833
Epoch 170: train loss 0.3846, dev loss 0.4540, train accuracy 0.8220, dev accuracy 0.7844
Epoch 175: train loss 0.3827, dev loss 0.4545, train accuracy 0.8224, dev accuracy 0.7833
Epoch 180: train loss 0.3810, dev loss 0.4550, train accuracy 0.8243, dev accuracy 0.7833
Epoch 185: train loss 0.3793, dev loss 0.4553, train accuracy 0.8256, dev accuracy 0.7833
Epoch 190: train loss 0.3776, dev loss 0.4561, train accuracy 0.8270, dev accuracy 0.7833
Epoch 195: train loss 0.3758, dev loss 0.4559, train accuracy 0.8273, dev accuracy 0.7833
Epoch 200: train loss 0.3740, dev loss 0.4565, train accuracy 0.8285, dev accuracy 0.7844
Metrics saved to Glove_DAN_metrics.csv
Curves saved to Glove_DAN_curves.png
Best recorded dev accuracy: 0.7913 at epoch 75
PS E:\OneDrive\Desktop\UCSD\CSE256 NLP\CSE256_PA1_FA26\CSE256_PA1_FA26> python main.py --model Embeddings_DAN --epochs 200
Read in 14923 vectors of size 300
Data loaded in : 0.8355963230133057 seconds
Epoch 5: train loss 0.6774, dev loss 0.6796, train accuracy 0.5621, dev accuracy 0.5482
Epoch 10: train loss 0.6595, dev loss 0.6613, train accuracy 0.6264, dev accuracy 0.6170
Epoch 15: train loss 0.6406, dev loss 0.6439, train accuracy 0.6486, dev accuracy 0.6353
Epoch 20: train loss 0.6156, dev loss 0.6261, train accuracy 0.6702, dev accuracy 0.6537
Epoch 25: train loss 0.5825, dev loss 0.6060, train accuracy 0.7095, dev accuracy 0.6674
Epoch 30: train loss 0.5424, dev loss 0.5864, train accuracy 0.7382, dev accuracy 0.6755
Epoch 35: train loss 0.4968, dev loss 0.5676, train accuracy 0.7733, dev accuracy 0.6915
Epoch 40: train loss 0.4473, dev loss 0.5523, train accuracy 0.8019, dev accuracy 0.7110
Epoch 45: train loss 0.3966, dev loss 0.5424, train accuracy 0.8311, dev accuracy 0.7190
Epoch 50: train loss 0.3463, dev loss 0.5376, train accuracy 0.8546, dev accuracy 0.7282
Epoch 55: train loss 0.2978, dev loss 0.5370, train accuracy 0.8818, dev accuracy 0.7317
Epoch 60: train loss 0.2523, dev loss 0.5411, train accuracy 0.9064, dev accuracy 0.7351
Epoch 65: train loss 0.2109, dev loss 0.5506, train accuracy 0.9260, dev accuracy 0.7466
Epoch 70: train loss 0.1742, dev loss 0.5670, train accuracy 0.9426, dev accuracy 0.7466
Epoch 75: train loss 0.1424, dev loss 0.5888, train accuracy 0.9574, dev accuracy 0.7489
Epoch 80: train loss 0.1155, dev loss 0.6136, train accuracy 0.9682, dev accuracy 0.7523
Epoch 85: train loss 0.0932, dev loss 0.6405, train accuracy 0.9770, dev accuracy 0.7580
Epoch 90: train loss 0.0749, dev loss 0.6694, train accuracy 0.9835, dev accuracy 0.7615
Epoch 95: train loss 0.0600, dev loss 0.6996, train accuracy 0.9882, dev accuracy 0.7557
Epoch 100: train loss 0.0481, dev loss 0.7301, train accuracy 0.9916, dev accuracy 0.7626
Epoch 105: train loss 0.0388, dev loss 0.7613, train accuracy 0.9938, dev accuracy 0.7638
Epoch 110: train loss 0.0315, dev loss 0.7911, train accuracy 0.9957, dev accuracy 0.7649
Epoch 115: train loss 0.0258, dev loss 0.8203, train accuracy 0.9968, dev accuracy 0.7638
Epoch 120: train loss 0.0214, dev loss 0.8491, train accuracy 0.9983, dev accuracy 0.7569
Epoch 125: train loss 0.0179, dev loss 0.8783, train accuracy 0.9987, dev accuracy 0.7546
Epoch 130: train loss 0.0152, dev loss 0.9046, train accuracy 0.9990, dev accuracy 0.7569
Epoch 135: train loss 0.0130, dev loss 0.9290, train accuracy 0.9991, dev accuracy 0.7580
Epoch 140: train loss 0.0112, dev loss 0.9515, train accuracy 0.9994, dev accuracy 0.7592
Epoch 145: train loss 0.0098, dev loss 0.9730, train accuracy 0.9994, dev accuracy 0.7603
Epoch 150: train loss 0.0086, dev loss 0.9938, train accuracy 0.9996, dev accuracy 0.7603
Epoch 155: train loss 0.0076, dev loss 1.0144, train accuracy 0.9997, dev accuracy 0.7592
Epoch 160: train loss 0.0067, dev loss 1.0325, train accuracy 0.9997, dev accuracy 0.7546
Epoch 165: train loss 0.0060, dev loss 1.0507, train accuracy 0.9997, dev accuracy 0.7557
Epoch 170: train loss 0.0054, dev loss 1.0684, train accuracy 0.9997, dev accuracy 0.7557
Epoch 175: train loss 0.0049, dev loss 1.0839, train accuracy 0.9997, dev accuracy 0.7569
Epoch 180: train loss 0.0044, dev loss 1.1001, train accuracy 0.9999, dev accuracy 0.7569
Epoch 185: train loss 0.0040, dev loss 1.1156, train accuracy 0.9999, dev accuracy 0.7557
Epoch 190: train loss 0.0036, dev loss 1.1290, train accuracy 0.9999, dev accuracy 0.7557
Epoch 195: train loss 0.0033, dev loss 1.1434, train accuracy 0.9999, dev accuracy 0.7580
Epoch 200: train loss 0.0031, dev loss 1.1575, train accuracy 0.9999, dev accuracy 0.7592
Metrics saved to Embeddings_DAN_metrics.csv
Curves saved to Embeddings_DAN_curves.png
Best recorded dev accuracy: 0.7649 at epoch 110

Training BPE on data/train.txt (target vocabulary 1000)...
BPE learned 932 merges; actual vocabulary 1000; time 29.41 seconds
train: mean subwords 35.18, max subwords 102
dev: mean subwords 35.73, max subwords 85
Data prepared in 38.18 seconds; batch size 64
Epoch 5: train loss 0.3917, dev loss 0.5397, train accuracy 0.8221, dev accuracy 0.7305
Epoch 10: train loss 0.2766, dev loss 0.5731, train accuracy 0.8952, dev accuracy 0.7305
Epoch 15: train loss 0.1705, dev loss 0.6493, train accuracy 0.9549, dev accuracy 0.7213
Epoch 20: train loss 0.0973, dev loss 0.7706, train accuracy 0.9815, dev accuracy 0.7236
Epoch 25: train loss 0.0474, dev loss 0.8806, train accuracy 0.9977, dev accuracy 0.7225
Epoch 30: train loss 0.0244, dev loss 1.0334, train accuracy 0.9997, dev accuracy 0.7144
Metrics saved to SUBWORDDAN_vocab1000_metrics.csv
Curves saved to SUBWORDDAN_vocab1000_curves.png
Best recorded dev accuracy: 0.7305 at epoch 5
Training and evaluation time: 12.84 seconds
Experiment settings and timing saved to SUBWORDDAN_vocab1000_summary.json
'''
