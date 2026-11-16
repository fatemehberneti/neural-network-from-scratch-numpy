"""
3D point classification with a neural network built from scratch in NumPy.

What is being classified?
-------------------------
Random points (x, y, z) are drawn uniformly from the cube [-10, 20]^3.
A "hat" surface (a downward-opening paraboloid) separates the space:

    z = 8 - (x - 3)^2 - (y - 5)^2

The surface has its peak at (x, y, z) = (3, 5, 8) and curves downward in every
direction. Each point gets a label depending on which side of the surface it
lies on:

    label = 1  if the point is ABOVE the surface  (z + (x-3)^2 + (y-5)^2 - 8 > 0)
    label = 0  if the point is BELOW the surface

The network only sees the raw (x, y, z) coordinates and has to learn this
curved boundary.

Architecture: 3 -> 10 -> 10 -> 10 -> 1, sigmoid activations, MSE loss,
full-batch gradient descent.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(40)

# Data generation

def generate_data(n_samples):
    # Uniform distribution: every value in [-10, 20] is equally likely.
    # Each row is one sample [x, y, z]; each column is one feature.
    # tozi yeknavakht / hame maqadir ehtemal barabar / pakhsh dar kole baze
    X = np.random.uniform(-10, 20, (n_samples, 3))

    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    # Signed "height" of the point relative to the hat surface.
    # Positive -> above the surface, negative -> below it.
    hat = z + (x - 3)**2 + (y - 5)**2 - 8

    y_label = (hat > 0).astype(int) # astype -> true =1 & false = 0
    y_label = y_label.reshape(n_samples, 1) # satri be sutuni , (m,) -> (m, 1) column vector

    return X, y_label


X_train, y_train = generate_data(10000)
X_test, y_test = generate_data(2000)

print(f"train data : {X_train.shape}, test data: {X_test.shape}")
print(f"class 0 : {np.sum(y_train==0)}, class 1: {np.sum(y_train==1)}\n")

# Activation function

def sigmoid(z):
    # Squashes any number into the range (0, 1), so the output can be read
    # as a probability.
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(s):
    # s is the OUTPUT of sigmoid (already activated), so sigma'(z) = s * (1 - s)
    return s * (1 - s)

# Neural network

class NeuralNetwork:
    def __init__(self, lr=0.1):
        # Weights ~ standard normal (mean 0, variance 1); biases start at 0.
        # W has shape (inputs, neurons): rows = inputs, columns = neurons.
        self.W1 = np.random.randn(3, 10) # 3 satr 10 sutun
        self.b1 = np.zeros((1, 10))

        self.W2 = np.random.randn(10, 10)
        self.b2 = np.zeros((1, 10))

        self.W3 = np.random.randn(10, 10)
        self.b3 = np.zeros((1, 10))

        self.W4 = np.random.randn(10, 1)
        self.b4 = np.zeros((1, 1))

        self.lr = lr # learning rate

    def forward(self, X):
        # Each sample is a row and the parameters are columns: z = X . W + b
        # (m, 3) · (3, 10) = (m, 10)
        self.z1 = np.dot(X, self.W1) + self.b1 # first layer input before activation / layer input before activation
        self.a1 = sigmoid(self.z1) # first layer output after activation / z1 -> sigmoid ruye har onsor = (0,1)
        
        # (m, 10) · (10, 10) = (m, 10)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        
        # (m, 10) · (10, 10) = (m, 10)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = sigmoid(self.z3)
        
        # (m, 10) · (10, 1) = (m, 1)
        self.z4 = np.dot(self.a3, self.W4) + self.b4
        self.a4 = sigmoid(self.z4)

        # Keep the input; it is needed in backpropagation.
        # zakhire vurudi brye estfde dr backpropagation
        self.X = X
        return self.a4

    def backward(self, y):
        m = y.shape[0] # number of samples

        # Output layer. Loss: L = 0.5 * (y_pred - y)^2
        # d4 = dL/da4 * da4/dz4 = (pred - y) * pred * (1 - pred)
        # گرادیان لایه خروجی - MSE: L = 0.5 * (y_pred - y)^2
        # error4 = dL/da4 * da4/dz4 * dz4/dw4 / vaqei - pred * pred (1-pred)
        d4 = (self.a4 - y) * sigmoid_derivative(self.a4)
        dW4 = np.dot(self.a3.T, d4) / m # a3 is the input of layer 4
        db4 = np.sum(d4, axis=0, keepdims=True) / m # /m = average gradient over all training samples
        
        # hidden layer 3
        d3 = np.dot(d4, self.W4.T) * sigmoid_derivative(self.a3)
        dW3 = np.dot(self.a2.T, d3) / m
        db3 = np.sum(d3, axis=0, keepdims=True) / m
        
        # hidden layer 2
        d2 = np.dot(d3, self.W3.T) * sigmoid_derivative(self.a2)
        dW2 = np.dot(self.a1.T, d2) / m
        db2 = np.sum(d2, axis=0, keepdims=True) / m
        
        # hidden layer 1
        d1 = np.dot(d2, self.W2.T) * sigmoid_derivative(self.a1)
        dW1 = np.dot(self.X.T, d1) / m
        db1 = np.sum(d1, axis=0, keepdims=True) / m

        # Gradient descent update
        self.W4 -= self.lr * dW4
        self.b4 -= self.lr * db4
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def compute_loss(self, y_true, y_pred):
        # Mean squared error
        return np.mean((y_pred - y_true) ** 2)

    def predict(self, X):
        probs = self.forward(X)
        return (probs >= 0.5).astype(int)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)


# Training (full-batch gradient descent)

nn = NeuralNetwork(lr=0.5)
n_epochs = 100

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for epoch in range(n_epochs):
    # forward pass
    y_pred_train = nn.forward(X_train)

    # loss
    train_loss = nn.compute_loss(y_train, y_pred_train)
    train_losses.append(train_loss)

    # Backpropagation + weight update
    nn.backward(y_train)

    # test loss
    y_pred_test = nn.forward(X_test)
    test_loss = nn.compute_loss(y_test, y_pred_test)
    test_losses.append(test_loss)
    
    # accuracy
    train_acc = nn.accuracy(X_train, y_train)
    train_accuracies.append(train_acc)
    test_acc = nn.accuracy(X_test, y_test)
    test_accuracies.append(test_acc)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"  Loss: Train={train_loss:.4f}, Test={test_loss:.4f}")
        print(f"  Accuracy: Train={train_acc:.4f}, Test={test_acc:.4f}\n")


# Confusion matrix, precision and recall (on the test set)
y_pred_test = nn.predict(X_test)

TP = np.sum((y_pred_test == 1) & (y_test == 1))
TN = np.sum((y_pred_test == 0) & (y_test == 0))
FP = np.sum((y_pred_test == 1) & (y_test == 0))
FN = np.sum((y_pred_test == 0) & (y_test == 1))

print("Confusion Matrix:")
print(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")

precision = TP / (TP + FP)
recall = TP / (TP + FN)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Plots: loss, accuracy, confusion matrix

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Loss curve
axes[0].plot(train_losses, label='Train', linewidth=2, color='purple')
axes[0].plot(test_losses, label='Test', linewidth=2, color='pink')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('MSE Loss', fontsize=12)
axes[0].set_title('Loss During Training', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Accuracy curve
axes[1].plot(train_accuracies, label='Train', linewidth=2, color='purple')
axes[1].plot(test_accuracies, label='Test', linewidth=2, color='pink')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Accuracy During Training', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

# confution matrix
confusion_matrix = np.array([[TN, FP], [FN, TP]])
im = axes[2].imshow(confusion_matrix, cmap='Purples', interpolation='nearest')
axes[2].set_xticks([0, 1])
axes[2].set_yticks([0, 1])
axes[2].set_xticklabels(['Pred 0', 'Pred 1'], fontsize=11)
axes[2].set_yticklabels(['True 0', 'True 1'], fontsize=11)
axes[2].set_title('Confusion Matrix', fontsize=14, fontweight='bold')

for i in range(2):
    for j in range(2):
        text = axes[2].text(j, i, str(confusion_matrix[i, j]), 
                           ha='center', va='center', fontsize=24, 
                           color='white' if confusion_matrix[i, j] > confusion_matrix.max()/2 else 'black',
                           fontweight='bold')

plt.colorbar(im, ax=axes[2])
plt.tight_layout()
plt.savefig('results.png', dpi=150, bbox_inches='tight')
plt.show()

print("Done !!!")