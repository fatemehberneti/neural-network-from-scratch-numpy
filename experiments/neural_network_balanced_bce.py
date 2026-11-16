import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data generation (balanced)
# -----------------------------
np.random.seed(40)

def generate_data(n_samples):
    X = np.random.uniform(-10, 20, (n_samples, 3))

    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    kollah = z + (x - 3)**2 + (y - 5)**2 - 8
    y_label = (kollah > 0).astype(int).reshape(-1, 1)

    return X, y_label


def generate_balanced_data(n_samples):
    X, y = generate_data(n_samples)

    X0 = X[y.flatten() == 0]
    X1 = X[y.flatten() == 1]
    y0 = y[y.flatten() == 0]
    y1 = y[y.flatten() == 1]

    n_min = min(len(X0), len(X1))

    idx0 = np.random.choice(len(X0), n_min, replace=False)
    idx1 = np.random.choice(len(X1), n_min, replace=False)

    X_bal = np.vstack([X0[idx0], X1[idx1]])
    y_bal = np.vstack([y0[idx0], y1[idx1]])

    perm = np.random.permutation(len(X_bal))
    return X_bal[perm], y_bal[perm]


X_train, y_train = generate_balanced_data(20000)
X_test, y_test = generate_balanced_data(5000)

print("Train shape:", X_train.shape)
print("Class 0:", np.sum(y_train == 0), "Class 1:", np.sum(y_train == 1))


# -----------------------------
# Activation
# -----------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# -----------------------------
# Neural Network
# -----------------------------
class NeuralNetwork:
    def __init__(self, lr=0.1):
        self.W1 = np.random.randn(3, 10) * 0.1
        self.b1 = np.zeros((1, 10))

        self.W2 = np.random.randn(10, 10) * 0.1
        self.b2 = np.zeros((1, 10))

        self.W3 = np.random.randn(10, 10) * 0.1
        self.b3 = np.zeros((1, 10))

        self.W4 = np.random.randn(10, 1) * 0.1
        self.b4 = np.zeros((1, 1))

        self.lr = lr

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)

        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = sigmoid(self.z3)

        self.z4 = self.a3 @ self.W4 + self.b4
        self.a4 = sigmoid(self.z4)

        self.X = X
        return self.a4

    def backward(self, y):
        m = y.shape[0]

        # BCE + sigmoid  --> gradient ساده میشه
        d4 = self.a4 - y
        dW4 = self.a3.T @ d4 / m
        db4 = np.mean(d4, axis=0, keepdims=True)

        d3 = (d4 @ self.W4.T) * self.a3 * (1 - self.a3)
        dW3 = self.a2.T @ d3 / m
        db3 = np.mean(d3, axis=0, keepdims=True)

        d2 = (d3 @ self.W3.T) * self.a2 * (1 - self.a2)
        dW2 = self.a1.T @ d2 / m
        db2 = np.mean(d2, axis=0, keepdims=True)

        d1 = (d2 @ self.W2.T) * self.a1 * (1 - self.a1)
        dW1 = self.X.T @ d1 / m
        db1 = np.mean(d1, axis=0, keepdims=True)

        self.W4 -= self.lr * dW4
        self.b4 -= self.lr * db4
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


# -----------------------------
# Loss & Metrics
# -----------------------------
def binary_cross_entropy(y, p):
    eps = 1e-8
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def precision_recall_f1(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1


# -----------------------------
# Training
# -----------------------------
nn = NeuralNetwork(lr=0.2)
epochs = 100

train_losses, test_losses = [], []
train_f1s, test_f1s = [], []

for epoch in range(epochs):
    # Train
    y_pred = nn.forward(X_train)
    loss = binary_cross_entropy(y_train, y_pred)
    nn.backward(y_train)

    # Test
    y_test_pred = nn.forward(X_test)
    test_loss = binary_cross_entropy(y_test, y_test_pred)

    # Metrics
    train_preds = (y_pred >= 0.5).astype(int)
    test_preds = (y_test_pred >= 0.5).astype(int)

    _, _, train_f1 = precision_recall_f1(y_train, train_preds)
    _, _, test_f1 = precision_recall_f1(y_test, test_preds)

    train_losses.append(loss)
    test_losses.append(test_loss)
    train_f1s.append(train_f1)
    test_f1s.append(test_f1)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}")
        print(f"  Loss: Train={loss:.4f}, Test={test_loss:.4f}")
        print(f"  F1  : Train={train_f1:.4f}, Test={test_f1:.4f}\n")


# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(train_losses, label="Train")
plt.plot(test_losses, label="Test")
plt.title("Binary Cross Entropy Loss")
plt.xlabel("Epoch")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1,2,2)
plt.plot(train_f1s, label="Train")
plt.plot(test_f1s, label="Test")
plt.title("F1 Score")
plt.xlabel("Epoch")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results_balanced.png', dpi=150, bbox_inches='tight')
plt.show()

print("Done!")