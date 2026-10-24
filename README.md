# CSE 256 PA1

Binary movie-review sentiment classification with word-level and subword-level
Deep Averaging Networks (DANs). The original Bag-of-Words (BOW) baseline is also
available. Part 3's Skip-Gram derivations are in the report and do not require a
separate program.

## 1. Setup and working directory

Use Python 3 with PyTorch, NumPy, scikit-learn, and Matplotlib:

```sh
python -m pip install torch numpy scikit-learn matplotlib
```

The report experiments used Python 3.13.14 and PyTorch 2.11.0+cpu. Training runs
on CPU; a GPU is not required. LaTeX is only needed to rebuild the report PDF.

Run the commands below **from the directory containing `main.py` and `data/`**.
The program uses relative data paths. Keep these supplied files in place:

```text
data/
    train.txt
    dev.txt
    glove.6B.300d-relativized.txt
```

Both word-level variants load this GloVe file. `Embeddings_DAN` uses only its
word index, then creates randomly initialized embeddings; it does not use the
pretrained vector values. `SUBWORDDAN` does not load GloVe.

## 2. Run each model

Model names are case-sensitive. These are the actual names supported by this
implementation; use `Glove_DAN` or `Embeddings_DAN` rather than `DAN`.

### Provided BOW baseline

```sh
python main.py --model BOW
```

Trains the supplied two-layer and three-layer BOW networks, each for 100 epochs
with batch size 16, and evaluates on dev. The DAN-specific epoch, batch-size,
vocabulary-size, and output-directory options do not change this baseline.

### Part 1a: DAN with frozen GloVe embeddings

```sh
python main.py --model Glove_DAN --epochs 200
```

Precomputes the mean 300-dimensional GloVe vector for each sentence and trains
the classifier. Embeddings remain fixed. Training is full-batch unless
`--batch-size` is supplied.

### Part 1b: DAN with randomly initialized, trainable embeddings

```sh
python main.py --model Embeddings_DAN --epochs 200
```

Converts sentences to padded word-ID sequences, looks up trainable random
embeddings inside the model, and averages over non-PAD tokens. Backpropagation
updates both the embedding matrix and classifier. This also defaults to
full-batch training; use `--batch-size 64` for mini-batches.

### Part 2: BPE subword DAN

```sh
python main.py --model SUBWORDDAN --vocab-size 1000 --epochs 30 --seed 42
python main.py --model SUBWORDDAN --vocab-size 3000 --epochs 30 --seed 42
python main.py --model SUBWORDDAN --vocab-size 5000 --epochs 30 --seed 42
```

Each invocation learns BPE from `data/train.txt`, encodes train and dev with the
same rules and vocabulary, then trains a DAN with random subword embeddings.
The default batch size for this model is 64. BPE fitting happens before the
first neural-network epoch and can take longer for larger vocabularies.

The vocabulary budget includes PAD, UNK, the initial characters, `</w>`, and
all newly introduced merged symbols. Earlier symbols are retained. Training
stops early if no adjacent pairs remain; the actual size is printed and saved.

## 3. Arguments and training settings

```sh
python main.py --help
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--model` | Required | `BOW`, `Glove_DAN`, `Embeddings_DAN`, or `SUBWORDDAN`. |
| `--epochs` | `200` | Training epochs for DAN models. |
| `--batch-size` | Model-dependent | Full batch for word-level DANs; 64 for `SUBWORDDAN`. Supply a positive integer to override. |
| `--vocab-size` | `1000` | Target BPE vocabulary size; applies only to `SUBWORDDAN`. |
| `--seed` | Unset | Optional PyTorch random seed. |
| `--output-dir` | `.` | Directory for DAN metrics, plots, and summaries. Created automatically. |

All DAN variants use 300-dimensional embeddings, a 100-unit hidden layer,
ReLU, dropout 0.2, and two output logits. They use Adam with learning rate
0.001 and `CrossEntropyLoss`. These architecture and optimizer settings are
defined in `DANmodels.py` and `main.py`; they are not command-line options.

Full-batch and mini-batch epochs contain different numbers of parameter
updates. To compare representations under the report's controlled protocol,
use the same batch size, seed, and epoch budget for every model:

```sh
python main.py --model Glove_DAN --epochs 30 --batch-size 64 --seed 42 --output-dir report_experiments
python main.py --model Embeddings_DAN --epochs 30 --batch-size 64 --seed 42 --output-dir report_experiments
python main.py --model SUBWORDDAN --vocab-size 1000 --epochs 30 --batch-size 64 --seed 42 --output-dir report_experiments
python main.py --model SUBWORDDAN --vocab-size 3000 --epochs 30 --batch-size 64 --seed 42 --output-dir report_experiments
python main.py --model SUBWORDDAN --vocab-size 5000 --epochs 30 --batch-size 64 --seed 42 --output-dir report_experiments
```

The timed report runs additionally fixed PyTorch to two CPU threads. To use
two threads when launching the commands above from PowerShell, first set:

```powershell
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
```

## 4. Outputs

DAN training prints train/dev cross-entropy and accuracy every five epochs,
and also at the final epoch if it is not a multiple of five. Both splits are
measured after the update in evaluation mode, with dropout disabled and no
gradients. The next training epoch restores training mode.

Each DAN run saves:

| File | Contents |
| --- | --- |
| `<name>_metrics.csv` | `epoch`, `train_loss`, `dev_loss`, `train_acc`, `dev_acc`. Accuracies are fractions in `[0, 1]`. |
| `<name>_curves.png` | Train/dev loss and accuracy plots; plotted accuracy is a percentage. |
| `<name>_summary.json` | Settings, final/best recorded dev accuracy, best recorded epoch, and preparation/training timings. BPE runs also record vocabulary and sequence-length statistics. |

`<name>` is `Glove_DAN`, `Embeddings_DAN`, or
`SUBWORDDAN_vocab<vocab-size>`. Different BPE sizes therefore have distinct
filenames. To keep multiple runs of the same configuration, give each run a
different `--output-dir`.

"Best recorded dev accuracy" is the maximum over the recorded evaluations.
The program records metrics, but does not save model weights or restore the
best epoch. BOW instead saves `train_accuracy.png` and `dev_accuracy.png` in
the working directory.

## 5. Changes relative to the starter code

| File | Changes |
| --- | --- |
| `main.py` | Added the three DAN routes; model-specific data preparation; configurable epochs, batch size, seed, BPE vocabulary, and output directory; shared train/dev monitoring; PNG/CSV output; JSON settings and timing summaries. BOW data preparation now runs only in the BOW branch. |
| `DANmodels.py` | Filled the originally empty file with the frozen-GloVe classifier, random-embedding DAN, word-level data conversion, PAD-masked averaging, and shared training/evaluation functions supporting full batches and mini-batches. |
| `bpe.py` (new) | Implemented frequency-weighted adjacent-pair merging, deterministic tie-breaking, ordered rule application, cached word encoding, unknown-token handling, and padded subword-ID datasets. BPE is fitted on train only. |
| `prepare_report.py` (new) | Generates the report's numerical LaTeX macros and comparison figures from saved CSV/JSON experiment results. Does not train models. |
| `Report.tex` | Added the model description, experimental results and analysis, figures/tables, references, and Part 3 derivations. |
| `build-report.ps1` (new) | Optional Windows helper that runs XeLaTeX twice and writes the report PDF to `build/`. |

`BOWmodels.py`, `sentiment_data.py`, and `utils.py` are unchanged from the
starter archive. The existing data reader, embedding loader, and `Indexer`
are reused. No changes to the supplied datasets or GloVe files are required.

## 6. Rebuild the report (optional)

With the supplied recorded results available:

```sh
python prepare_report.py
```

This requires the original word-model metric CSVs in the project directory,
the five controlled-run metric CSVs and summary JSONs in `report_experiments/`,
and its `environment.json`. It generates `report_experiments/report_numbers.tex`
and the report figures. The recorded results are included separately from the
training code.

On Windows with MiKTeX installed:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build-report.ps1
```

The PDF is `build/Report.pdf`. Keep the generated number macros and figures
alongside `Report.tex` when compiling the report.
