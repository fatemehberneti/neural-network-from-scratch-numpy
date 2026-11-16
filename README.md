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

## Results

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

Ideas to fix this:

- Balance the classes by undersampling the majority class.
- Evaluate with precision, recall and F1 instead of accuracy.
- Use binary cross-entropy instead of MSE for a sigmoid output.
- Normalize or scale the inputs.

## Project structure

```
├── neural_network.py   # data generation, network, training, evaluation, plots
├── requirements.txt
├── results.png
└── README.md
```
