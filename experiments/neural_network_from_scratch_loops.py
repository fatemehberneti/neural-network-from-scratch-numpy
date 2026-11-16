import numpy as np
import matplotlib.pyplot as plt

# This is the sigmoid function
# It takes any number and squashes it between 0 and 1
def sigmoid(x):
    # We need to clip x to avoid overflow errors
    # If x is too big, exp(-x) becomes 0, which is fine
    # If x is too small, exp(-x) becomes huge, so we limit it
    if x > 500:
        x = 500
    if x < -500:
        x = -500
    result = 1.0 / (1.0 + np.exp(-x))
    return result

# This is the derivative of sigmoid
# We need this for backpropagation
def sigmoid_derivative(x):
    # The derivative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x))
    s = sigmoid(x)
    return s * (1.0 - s)

# Generate training and test data
# We need to create points in 3D space and label them
def generate_data(num_samples):
    # The formula is: 8 = (y-5)^2 + z + (x-3)^2
    # If (y-5)^2 + z + (x-3)^2 > 8, then label = 1 (above hat)
    # If (y-5)^2 + z + (x-3)^2 < 8, then label = 0 (below hat)
    
    # Create arrays to store x, y, z coordinates
    x_coords = []
    y_coords = []
    z_coords = []
    labels = []
    
    # Generate random points
    for i in range(num_samples):
        # Random number between -10 and 20 for each coordinate
        x = np.random.uniform(-10, 20)
        y = np.random.uniform(-10, 20)
        z = np.random.uniform(-10, 20)
        
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)
        
        # Calculate left side of equation: (y-5)^2 + z + (x-3)^2
        term1 = (y - 5) * (y - 5)  # (y-5)^2
        term2 = z
        term3 = (x - 3) * (x - 3)  # (x-3)^2
        left_side = term1 + term2 + term3
        
        # Label: 1 if above hat, 0 if below hat
        if left_side > 8:
            labels.append(1)
        else:
            labels.append(0)
    
    # Combine x, y, z into one array
    data = np.array([x_coords, y_coords, z_coords]).T
    
    return data, np.array(labels)


# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        # input_size = 3 (x, y, z coordinates)
        # hidden_sizes = [10, 10, 10] (three hidden layers with 10 neurons each)
        # output_size = 1 (one output: 0 or 1)
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Store all layer sizes in a list
        # Example: [3, 10, 10, 10, 1]
        self.layers = [input_size]
        for size in hidden_sizes:
            self.layers.append(size)
        self.layers.append(output_size)
        
        # Initialize weights and biases
        # Weights connect neurons between layers
        # Biases are added to each neuron
        self.weights = []
        self.biases = []
        
        # For each pair of consecutive layers, create weight matrix and bias vector
        for i in range(len(self.layers) - 1):
            # Weight matrix: rows = neurons in next layer, cols = neurons in current layer
            # Initialize with small random values
            w = np.random.randn(self.layers[i+1], self.layers[i]) * 0.5
            # Bias vector: one bias for each neuron in next layer
            b = np.random.randn(self.layers[i+1], 1) * 0.5
            
            self.weights.append(w)
            self.biases.append(b)
    
    # Feedforward: calculate output from input
    def feed_forward(self, x):
        # x is input vector, shape should be (3, 1) for our case
        # Make sure x is a column vector
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        
        # Store activations for each layer
        # activations[0] = input layer
        # activations[1] = first hidden layer
        # activations[2] = second hidden layer
        # activations[3] = third hidden layer
        # activations[4] = output layer
        activations = [x]
        
        # Store z values (before sigmoid) for each layer
        # We need these for backpropagation
        z_values = []
        
        # Go through each layer
        for i in range(len(self.weights)):
            # Calculate z = W * a + b
            # W is weight matrix, a is activation from previous layer, b is bias
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            z_values.append(z)
            
            # Apply sigmoid activation function
            # We need to apply sigmoid element-wise to the vector z
            a = np.zeros_like(z)
            for j in range(z.shape[0]):
                a[j, 0] = sigmoid(z[j, 0])
            
            activations.append(a)
        
        return activations, z_values
    
    # Backpropagation: calculate gradients
    def backpropagation(self, x, y):
        # x is input, y is target output
        # Make sure x and y are column vectors
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if isinstance(y, (int, float)):
            y = np.array([[float(y)]])
        elif len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # Step 1: Feedforward to get activations and z values
        activations, z_values = self.feed_forward(x)
        
        # Step 2: Calculate error in output layer
        # delta_L = (a_L - y) * sigmoid_derivative(z_L)
        output_activation = activations[-1]
        output_z = z_values[-1]
        
        # Calculate derivative of sigmoid for output layer
        sigmoid_deriv_output = np.zeros_like(output_z)
        for i in range(output_z.shape[0]):
            sigmoid_deriv_output[i, 0] = sigmoid_derivative(output_z[i, 0])
        
        # Calculate delta for output layer
        delta_output = (output_activation - y) * sigmoid_deriv_output
        deltas = [delta_output]
        
        # Step 3: Backpropagate error to hidden layers
        # Go backwards through layers
        for i in range(len(self.weights) - 1, 0, -1):
            # delta_l = (W_{l+1}^T * delta_{l+1}) * sigmoid_derivative(z_l)
            # Transpose weight matrix and multiply by delta from next layer
            delta_next = deltas[0]
            w_transpose = self.weights[i].T
            delta_prev = np.dot(w_transpose, delta_next)
            
            # Multiply by sigmoid derivative
            z_prev = z_values[i-1]
            sigmoid_deriv_prev = np.zeros_like(z_prev)
            for j in range(z_prev.shape[0]):
                sigmoid_deriv_prev[j, 0] = sigmoid_derivative(z_prev[j, 0])
            
            delta_prev = delta_prev * sigmoid_deriv_prev
            deltas.insert(0, delta_prev)
        
        # Step 4: Calculate gradients
        # Gradient for weights: delta * activation^T
        # Gradient for biases: delta
        weight_gradients = []
        bias_gradients = []
        
        for i in range(len(self.weights)):
            # Weight gradient = delta * activation^T
            weight_grad = np.dot(deltas[i], activations[i].T)
            # Bias gradient = delta
            bias_grad = deltas[i]
            
            weight_gradients.append(weight_grad)
            bias_gradients.append(bias_grad)
        
        return weight_gradients, bias_gradients
    
    # Train the network
    def train(self, X_train, y_train, epochs, learning_rate):
        # X_train: training data, shape (num_samples, 3)
        # y_train: training labels, shape (num_samples,)
        # epochs: number of training iterations
        # learning_rate: how big steps we take in gradient descent
        
        num_samples = X_train.shape[0]
        loss_history = []
        
        print("Starting training...")
        print("Number of training samples:", num_samples)
        print("Network structure:", self.layers)
        print("Number of epochs:", epochs)
        print("Learning rate:", learning_rate)
        print("-" * 60)
        
        # Train for specified number of epochs
        for epoch in range(epochs):
            # Initialize total gradients (we'll sum gradients from all samples)
            total_weight_grads = []
            total_bias_grads = []
            for i in range(len(self.weights)):
                total_weight_grads.append(np.zeros_like(self.weights[i]))
                total_bias_grads.append(np.zeros_like(self.biases[i]))
            
            total_loss = 0.0
            
            # Process all training samples (full batch)
            for sample_idx in range(num_samples):
                # Get one training sample
                x = X_train[sample_idx]
                y = y_train[sample_idx]
                
                # Calculate gradients using backpropagation
                weight_grads, bias_grads = self.backpropagation(x, y)
                
                # Add gradients to total (we'll average later)
                for i in range(len(self.weights)):
                    total_weight_grads[i] = total_weight_grads[i] + weight_grads[i]
                    total_bias_grads[i] = total_bias_grads[i] + bias_grads[i]
                
                # Calculate loss (mean squared error)
                activations, _ = self.feed_forward(x)
                output = activations[-1][0, 0]
                target = float(y)
                error = (output - target) ** 2
                total_loss = total_loss + error
            
            # Average the gradients (divide by number of samples)
            for i in range(len(self.weights)):
                avg_weight_grad = total_weight_grads[i] / num_samples
                avg_bias_grad = total_bias_grads[i] / num_samples
                
                # Update weights and biases using gradient descent
                # new_weight = old_weight - learning_rate * gradient
                self.weights[i] = self.weights[i] - learning_rate * avg_weight_grad
                self.biases[i] = self.biases[i] - learning_rate * avg_bias_grad
            
            # Calculate average loss
            avg_loss = total_loss / num_samples
            loss_history.append(avg_loss)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print("Epoch", epoch + 1, "/", epochs, "- Loss:", avg_loss)
        
        print("-" * 60)
        print("Training finished!")
        
        return loss_history
    
    # Predict label for a single input
    def predict(self, x):
        # Get output from network
        activations, _ = self.feed_forward(x)
        output = activations[-1][0, 0]
        
        # If output > 0.5, predict 1, else predict 0
        if output > 0.5:
            return 1
        else:
            return 0
    
    # Evaluate accuracy on test data
    def evaluate(self, X_test, y_test):
        num_samples = X_test.shape[0]
        correct = 0
        
        # Check each test sample
        for i in range(num_samples):
            prediction = self.predict(X_test[i])
            if prediction == y_test[i]:
                correct = correct + 1
        
        # Calculate accuracy percentage
        accuracy = (correct / num_samples) * 100.0
        return accuracy


# Main program
if __name__ == "__main__":
    # Set random seed so results are reproducible
    np.random.seed(42)
    
    # Generate training data (10,000 samples)
    print("Generating training data...")
    X_train, y_train = generate_data(10000)
    print("Training data shape:", X_train.shape)
    print("Training labels shape:", y_train.shape)
    
    # Generate test data (2,000 samples)
    print("Generating test data...")
    X_test, y_test = generate_data(2000)
    print("Test data shape:", X_test.shape)
    print("Test labels shape:", y_test.shape)
    print()
    
    # Create neural network
    # 3 inputs (x, y, z)
    # 3 hidden layers with 10 neurons each
    # 1 output (0 or 1)
    print("Creating neural network...")
    nn = NeuralNetwork(input_size=3, hidden_sizes=[10, 10, 10], output_size=1)
    print()
    
    # Train the network
    # 100 epochs, learning rate 0.1
    loss_history = nn.train(X_train, y_train, epochs=100, learning_rate=0.1)
    print()
    
    # Test the network
    print("Testing on test data...")
    test_accuracy = nn.evaluate(X_test, y_test)
    print("Test accuracy:", test_accuracy, "%")
    print()
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color='purple', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Loss curve saved.")
    
    # Show some example predictions
    print("=" * 60)
    print("Example Predictions:")
    print("=" * 60)
    for i in range(5):
        x_val = X_test[i][0]
        y_val = X_test[i][1]
        z_val = X_test[i][2]
        true_label = y_test[i]
        pred_label = nn.predict(X_test[i])
        
        print("Sample", i + 1, ":")
        print("  Position: x =", round(x_val, 2), ", y =", round(y_val, 2), ", z =", round(z_val, 2))
        print("  True label:", true_label, "| Prediction:", pred_label, end="")
        if true_label == pred_label:
            print(" | Correct!")
        else:
            print(" | Wrong!")
        print()
