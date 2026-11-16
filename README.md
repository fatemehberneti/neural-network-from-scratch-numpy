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

## Known limitation: class imbalance

Because most of the cube lies above the hat, the classes are very unbalanced (about 98% class 1 and 2% class 0 in the training set). With this data the network learns to predict class 1 for almost every point. That gives roughly 98% accuracy, but the confusion matrix shows `TN = 0`: it never correctly identifies a point below the surface. **Accuracy alone is misleading here.**

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
