# ECS708U Fashion-MNIST Coursework

This repository contains everything needed to reproduce the coursework: the notebook, the saved data splits
and seeds, the trained model checkpoints, the cached features, the experiment configurations and the
software versions.

The work is in one notebook, `ECS708U_P_Fashion_MNIST_Coursework.ipynb`, organised in parts:

| Part | Topic |
|---|---|
| 0 | Fixed data partitions (train / validation / test, clustering subsets) |
| 1 | Classical classification: logistic regression and kNN |
| 2 | Neural networks: baseline MLP, capacity and learning-rate experiments |
| 3 | Overfitting, regularisation (dropout, weight decay) and randomness (3 seeds) |
| 4 | Clustering: K-means and hierarchical clustering on raw pixels |
| 5 | MLP features vs raw pixels, and Route B (ensembling) |
| 6 | Final synthesis |

---

## 1. Setup and installation

### Option A: Google Colab (how the results were produced)

1. Open `ECS708U_P_Fashion_MNIST_Coursework.ipynb` in Google Colab.
2. Keep the default **CPU** runtime (**Runtime → Change runtime type → CPU**).
3. Colab already includes all the libraries used.
4. Run **Runtime → Run all**.

### Option B: your own machine

```bash
git clone https://github.com/butthamza20/ECS708U-P-Fashion-MNIST-Coursework.git
cd ECS708U-P-Fashion-MNIST-Coursework
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook ECS708U_P_Fashion_MNIST_Coursework.ipynb
```

Then run all cells from top to bottom.

### Data

No manual download is needed. The first cells download Fashion-MNIST [1] with `torchvision.datasets.FashionMNIST`
into a local folder called `data/` (relative path). This folder is not committed to the repository.

---

## 2. Running the experiments

All experiments run inside the notebook, in order, from top to bottom. There are no separate scripts.

| To reproduce | Run |
|---|---|
| Everything | **Run all** (Colab) or **Cell → Run all** (Jupyter) |
| Without opening the notebook | `jupyter nbconvert --to notebook --execute ECS708U_P_Fashion_MNIST_Coursework.ipynb --output executed.ipynb` |

Each experiment ID in `configs/experiments.json` names the notebook section that produces it.

**Important:** cells must be run in order. Later parts reuse objects created earlier (for example, Part 5 uses the
Part 2 checkpoint and the Part 4 clustering settings).

---

## 3. Hardware and approximate runtimes

All results were produced on a **CPU** (Google Colab, no GPU), with **Python 3.13.18**, **PyTorch 2.14.1**,
**Torchvision 0.29.1**, **NumPy 2.4.6**, **scikit-learn 1.6.1**, **SciPy 1.16.3**, **pandas 2.2.3** and
**Matplotlib 3.10.0**. Runtimes vary between Colab sessions, so they are approximate.

| Step | Time in the final run |
|---|---|
| Logistic regression, C = 1, 50,000 images (fit) | 291.8 s |
| Logistic regression, C = 0.01, 50,000 images (fit) | 155 s |
| kNN prediction, 10,000 validation images | 28.9 s |
| kNN k sweep (8 values) | 43–52 s per value |
| Logistic regression C sweep (7 values, 10,000-image subset) | 11–112 s per value |
| Baseline MLP, 20 epochs | 172 s |
| Small / large MLP, 20 epochs | 146 s / 321 s |
| MLP lr 0.01 / lr 0.5, 20 epochs | 172 s / 172 s |
| Part 3 regularisation runs, 60 epochs on 10,000 images | 50–58 s per run (9 runs) |
| K-means K sweep (12 values, raw pixels) | 1.2–6.0 s per value |
| K-means K sweep (12 values, MLP features) | 8 s in total |
| Ward linkage, 300 images | 0.03 s |
| MLP feature extraction, 4,000 images | 0.06 s |
| Ensemble prediction (logistic regression + MLP), 10,000 images | 0.14 s |

**Full notebook ("Run all"):** about **1 hour** on a Colab CPU. The timed steps in this table add up to about
56 minutes; a few steps (for example the K-means stability checks) were not timed separately.

---

## 4. Seeds and saved indices

**Seed:** `SEED = 42` is set once in the setup cell (`random`, `numpy`, `torch`) and used for every split,
subset and default training run. Part 3 also uses seeds **42, 43 and 44**, and the Part 4 stability check uses
seeds **0–9**.

All indices are saved as NumPy arrays in `splits/` and can be loaded with `np.load(...)`.

| File | Contents | Size | Index convention |
|---|---|---|---|
| `splits/train_indices.npy` | training partition | 50,000 | positions in the official 60,000 training images |
| `splits/val_indices.npy` | validation partition | 10,000 | positions in the official 60,000 training images |
| `splits/cluster_train_indices.npy` | clustering train subset (300 per class) | 3,000 | positions in the official 60,000 training images |
| `splits/cluster_val_indices.npy` | clustering validation subset (100 per class) | 1,000 | positions in the official 60,000 training images |
| `splits/cluster_test_indices.npy` | clustering test subset (100 per class) | 1,000 | positions in the official 10,000 test images |
| `splits/reg_train_indices.npy` | Part 3 training subset (1,000 per class) | 10,000 | positions in the official 60,000 training images |
| `splits/hier_indices.npy` | Part 4.4 hierarchical subset (30 per class) | 300 | positions in the official 60,000 training images |

All splits and subsets are stratified by class and created with `sklearn.model_selection.train_test_split`
(`random_state = 42`). Labels are used only to balance the subsets, never to fit clusters.

Two further samples are created in the notebook with seed 42 but not saved as files: the 5,000-row kNN training
sample (rows of the 50,000-image training array, `np.random.default_rng(42)`) and the 10,000-row logistic
regression C-sweep subset (rows of the 50,000-image training array).

---

## 5. Configurations and experiment IDs

`configs/experiments.json` lists every experiment (EXP-01 to EXP-19): the question, the variable changed,
the fixed settings, the seed(s) and the notebook section. The same register appears in the notebook.

### Saved checkpoints and features

| File | What it is |
|---|---|
| `checkpoints/baseline.pt` | Baseline MLP [256, 128], best validation epoch 16 (the validation-selected MLP) |
| `checkpoints/small.pt`, `checkpoints/large.pt` | Capacity experiment (Part 2.2) |
| `checkpoints/lr_0.01.pt`, `checkpoints/lr_0.5.pt` | Learning-rate experiment (Part 2.3) |
| `checkpoints/none_s42.pt`, `none_s43.pt`, `none_s44.pt` | Part 3, no regularisation, 3 seeds |
| `checkpoints/dropout_0.2_s42.pt`, `_s43.pt`, `_s44.pt` | Part 3, dropout 0.2, 3 seeds |
| `checkpoints/dropout_0.5_s42.pt`, `wd_1e-4_s42.pt`, `wd_1e-3_s42.pt` | Part 3 exploration runs |
| `features/mlp_hidden2_train.npy`, `features/mlp_hidden2_val.npy` | 128-d features from the last hidden layer of `baseline.pt`, for the clustering train and validation subsets |

All checkpoints are PyTorch `state_dict` files saved with `torch.save` and loaded with `torch.load`.

---

## 6. Code imported by the notebook

The notebook imports **no local modules**: all code is inside the notebook. It uses only the libraries in
`requirements.txt`.

---

## 7. Repository layout

```
ECS708U-P-Fashion-MNIST-Coursework/
├── ECS708U_P_Fashion_MNIST_Coursework.ipynb   # all code, results and write-ups
├── README.md                                  # this file
├── requirements.txt                           # pinned library versions
├── configs/
│   └── experiments.json                       # experiment IDs and settings
├── splits/                                    # saved split and subset indices
├── checkpoints/                               # trained MLP weights
└── features/                                  # cached MLP features
```

`data/` (the downloaded dataset) is excluded by `.gitignore`.

---

## 8. Acknowledgements

**Dataset**
- Fashion-MNIST, by Zalando Research [1].

**Libraries**
- PyTorch and Torchvision: data loading, MLP training.
- scikit-learn: data splitting, scaling, logistic regression, kNN, K-means and metrics.
- SciPy: hierarchical clustering.
- NumPy, pandas, Matplotlib: arrays, tables and figures.

**Course material**
- Module lectures 05-Classification-II, 06-NeuralNetworks and 07-Clustering (I. Patras; slide acknowledgements
  to T. Hospedales) [2].

**External code and checkpoints**
- No external code was copied, and no pre-trained checkpoints were used. All checkpoints in `checkpoints/` were
  trained from scratch in this notebook.

**Assistance**
- Claude was used for guidance on code structure, debugging and drafting explanations. All code was run, checked and understood by me, and all
  results come from my own runs."]

---

## References

[1] Zalando Research, Fashion-MNIST repository and documentation, https://github.com/zalandoresearch/fashion-mnist.

[2] Module lectures: 05-Classification-II, 06-NeuralNetworks, 07-Clustering (I. Patras; slide acknowledgements to
T. Hospedales).
