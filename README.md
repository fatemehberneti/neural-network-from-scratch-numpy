# 3D Point Classification with a Neural Network from Scratch (NumPy)

A small neural network built **only with NumPy** (no TensorFlow or PyTorch) that classifies points in 3D space as lying above or below a hat-shaped surface. Forward pass, backpropagation and gradient descent are all implemented by hand.

## What is being classified?

Random points `(x, y, z)` are sampled uniformly from the cube `[-10, 20]³`. The surface

```
z = 8 - (x - 3)² - (y - 5)²
```

is a downward-opening paraboloid (a "hat") with its peak at `(3, 5, 8)`. Each point is labeled by which side of the surface it is on:

| Label | Meaning | Condition |
|-------|---------|-----------|
| 1 | above the surface | `z + (x-3)² + (y-5)² - 8 > 0` |
| 0 | below the surface | otherwise |

The network only sees the raw coordinates and has to learn the curved boundary.

## Model

| Item | Value |
|------|-------|
| Architecture | 3 → 10 → 10 → 10 → 1 |
| Activation | Sigmoid (all layers) |
| Loss | Mean squared error (MSE) |
| Optimizer | Full-batch gradient descent, learning rate 0.5 |
| Epochs | 100 |
| Data | 10,000 training / 2,000 test samples |
| Weight init | Standard normal, biases zero |

## Run it

```bash
pip install -r requirements.txt
python neural_network.py
```

The script prints loss and accuracy every 20 epochs, then the confusion matrix, precision and recall. It saves the plots to `results.png`.

## Results (main model)

![Results](results.png)

The plots show the loss curve, the accuracy curve and the confusion matrix on the test set.

Dataset: 10,000 training samples (207 of class 0, 9,793 of class 1) and 2,000 test samples.

| Epoch | Train loss | Test loss | Train accuracy | Test accuracy |
|-------|-----------|-----------|----------------|---------------|
| 20    | 0.0367    | 0.0331    | 0.9793         | 0.9840        |
| 40    | 0.0257    | 0.0222    | 0.9793         | 0.9840        |
| 60    | 0.0230    | 0.0194    | 0.9793         | 0.9840        |
| 80    | 0.0219    | 0.0182    | 0.9793         | 0.9840        |
| 100   | 0.0214    | 0.0175    | 0.9793         | 0.9840        |

Confusion matrix on the test set (2,000 samples):

|            | Pred 0 | Pred 1 |
|------------|--------|--------|
| **True 0** | TN = 0 | FP = 32 |
| **True 1** | FN = 0 | TP = 1968 |

- Precision: **0.9840**
- Recall: **1.0000**
- Test accuracy: **98.40%**

The loss keeps decreasing, but accuracy is flat from about epoch 10 onward. That is a sign the model has settled on predicting one class for everything (see the limitation below).

## Known limitation: class imbalance

Because most of the cube lies above the hat, the classes are very unbalanced (207 vs 9,793 in the training set, about 2% vs 98%). With this data the network learns to predict class 1 for every point. That gives 98.40% test accuracy, but the confusion matrix shows `TN = 0`: it never correctly identifies a point below the surface. A model that always answers "1" would score the same accuracy, so **accuracy alone is misleading here.** The 100% recall and 98.4% precision only reflect that the test set is almost entirely class 1.

## Experiments

Two additional scripts in `experiments/` explore the same problem in different ways.

### 1. Loop-based version (`neural_network_from_scratch_loops.py`)

A "learning version" that processes **one sample at a time** with plain Python loops instead of matrix operations. It uses the same 3 → 10 → 10 → 10 → 1 sigmoid network with MSE loss and full-batch gradient descent (gradients are summed over all samples, then averaged), learning rate 0.1, 100 epochs, on the same unbalanced data distribution.

```bash
python experiments/neural_network_from_scratch_loops.py
```

Note: because of the Python loops it is much slower than the vectorized version. Reduce `epochs` in the script for a quick demo.

| Epoch | Average MSE loss |
|-------|------------------|
| 1     | 0.5316 |
| 10    | 0.3774 |
| 20    | 0.2375 |
| 40    | 0.1082 |
| 60    | 0.0666 |
| 80    | 0.0493 |
| 100   | 0.0405 |

**Test accuracy: 98.45%.** The five example predictions printed by the script were all class 1 and all correct. This is almost exactly the share of class 1 in the data, so this result is consistent with the same imbalance problem as the main model (predicting class 1 almost everywhere). The script does not print a confusion matrix, so this cannot be confirmed from its output alone.

### 2. Balanced data + binary cross-entropy (`neural_network_balanced_bce.py`)

An attempt to fix the imbalance: the majority class is undersampled so both classes have the same size, the loss is binary cross-entropy (which pairs naturally with a sigmoid output), and the metric is F1 score instead of accuracy. Weights start small (`randn * 0.1`) and the learning rate is 0.2.

```bash
python experiments/neural_network_balanced_bce.py
```

Undersampling leaves very little data: generating 20,000 points gives only **744 training samples** (372 per class).

| Epoch | Train BCE | Test BCE | Train F1 | Test F1 |
|-------|-----------|----------|----------|---------|
| 20    | 0.6931    | 0.6931   | 0.6667   | 0.6667  |
| 40    | 0.6931    | 0.6931   | 0.5475   | 0.6460  |
| 60    | 0.6931    | 0.6931   | 0.5719   | 0.6515  |
| 80    | 0.6931    | 0.6931   | 0.6266   | 0.7681  |
| 100   | 0.6931    | 0.6931   | 0.6738   | 0.8276  |

**This experiment did not train successfully.** A BCE loss of 0.6931 is `ln(2)`, the loss of a model that outputs 0.5 for every input, and it never moved from that value. The F1 score changes between epochs only because the outputs sit right around the 0.5 threshold, so small shifts flip many predictions. It is not a sign of learning, and the 0.83 test F1 should not be read as a good result.

Likely causes and things to try:

- Small initial weights plus four sigmoid layers lead to very small gradients (vanishing gradients) in the early layers.
- The inputs are unscaled (range -10 to 20), which makes the optimization harder. Standardize them.
- Only 744 training samples. Generate more points before balancing (for example 200,000).
- Use a better initialization (Xavier/He), ReLU hidden layers, or train for more epochs.

## Project structure

```
├── neural_network.py                          # main model: data, network, training, evaluation, plots
├── experiments/
│   ├── neural_network_from_scratch_loops.py   # slow, per-sample loop version
│   └── neural_network_balanced_bce.py         # balanced data + BCE + F1 (did not converge)
├── requirements.txt
├── results.png
└── README.md
```
