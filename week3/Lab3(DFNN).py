import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Set seed for reproducibility
np.random.seed(42)

# =====================================================================
# 1. LOAD & EXPLORE DATASET
# =====================================================================
print("Loading Covertype dataset...")
X_raw, y_raw = fetch_covtype(return_X_y=True)

# Convert target from 1-7 to 0-6 indexing
y_zero_indexed = y_raw - 1
num_classes = 7
num_samples, num_features = X_raw.shape

print(f"Dataset shape: {X_raw.shape}, Classes: {num_classes}")

# Analyze class distribution & imbalance ratio
class_counts = np.bincount(y_zero_indexed)
max_class_count = np.max(class_counts)
min_class_count = np.min(class_counts)
imbalance_ratio = max_class_count / min_class_count

print("\n--- Class Distribution ---")
for i, count in enumerate(class_counts):
    print(f"Class {i}: {count} samples ({count / num_samples * 100:.2f}%)")
print(f"Imbalance Ratio (Max / Min): {imbalance_ratio:.2f}\n")

# One-hot encoding target
Y_onehot = np.eye(num_classes)[y_zero_indexed]

# =====================================================================
# 2. PARTITION & PREPROCESS DATA
# =====================================================================
# 80% train, 10% validation, 10% test with stratified splits
X_train, X_temp, Y_train_oh, Y_temp_oh, y_train, y_temp = train_test_split(
    X_raw, Y_onehot, y_zero_indexed, test_size=0.20, random_state=42, stratify=y_zero_indexed
)

X_val, X_test, Y_val_oh, Y_test_oh, y_val, y_test = train_test_split(
    X_temp, Y_temp_oh, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Standardize features (fit only on training set)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Calculate class weights: w_c = N / (K * n_c)
N_train = len(y_train)
class_weights = N_train / (num_classes * class_counts)
print(f"Calculated Class Weights: {np.round(class_weights, 3)}")

# =====================================================================
# 3. CUSTOM MLP ARCHITECTURE FROM SCRATCH
# =====================================================================
class NeuralNetworkFromScratch:
    def __init__(self, layer_dims, class_weights):
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        self.class_weights = class_weights
        self.params = {}
        self._initialize_parameters()

    def _initialize_parameters(self):
        # He (Kaiming) Initialization for ReLU hidden layers
        for l in range(1, self.num_layers + 1):
            n_in = self.layer_dims[l - 1]
            n_out = self.layer_dims[l]
            self.params[f'W{l}'] = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            self.params[f'b{l}'] = np.zeros((1, n_out))

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def relu_derivative(Z):
        return (Z > 0).astype(float)

    @staticmethod
    def softmax(Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        caches = {}
        A = X
        caches['A0'] = A
        
        # Hidden layers with ReLU
        for l in range(1, self.num_layers):
            Z = np.dot(A, self.params[f'W{l}']) + self.params[f'b{l}']
            A = self.relu(Z)
            caches[f'Z{l}'] = Z
            caches[f'A{l}'] = A

        # Output layer with Softmax
        Z_out = np.dot(A, self.params[f'W{self.num_layers}']) + self.params[f'b{self.num_layers}']
        A_out = self.softmax(Z_out)
        caches[f'Z{self.num_layers}'] = Z_out
        caches[f'A{self.num_layers}'] = A_out

        return A_out, caches

    def compute_weighted_loss(self, Y_hat, Y_true):
        # Y_true is one-hot encoded
        m = Y_true.shape[0]
        # Sample weights based on target class
        sample_weights = np.sum(Y_true * self.class_weights, axis=1, keepdims=True)
        
        eps = 1e-15  # Numerical stability clipping
        Y_hat_clipped = np.clip(Y_hat, eps, 1 - eps)
        
        loss_per_sample = -np.sum(Y_true * np.log(Y_hat_clipped), axis=1, keepdims=True)
        weighted_loss = np.mean(loss_per_sample * sample_weights)
        return weighted_loss

    def backward(self, caches, Y_true):
        grads = {}
        m = Y_true.shape[0]
        
        # Sample weights for backward pass
        sample_weights = np.sum(Y_true * self.class_weights, axis=1, keepdims=True)
        
        # Gradient at output layer (Softmax + Cross Entropy with sample weights)
        A_last = caches[f'A{self.num_layers}']
        dZ = (A_last - Y_true) * sample_weights / m

        for l in range(self.num_layers, 0, -1):
            A_prev = caches[f'A{l-1}']
            grads[f'dW{l}'] = np.dot(A_prev.T, dZ)
            grads[f'db{l}'] = np.sum(dZ, axis=0, keepdims=True)

            if l > 1:
                dA_prev = np.dot(dZ, self.params[f'W{l}'].T)
                dZ = dA_prev * self.relu_derivative(caches[f'Z{l-1}'])

        return grads

    def update_parameters(self, grads, lr):
        for l in range(1, self.num_layers + 1):
            self.params[f'W{l}'] -= lr * grads[f'dW{l}']
            self.params[f'b{l}'] -= lr * grads[f'db{l}']

# =====================================================================
# 4. MODEL TRAINING & EARLY STOPPING
# =====================================================================
layer_dims = [54, 64, 32, 16, 7]
model = NeuralNetworkFromScratch(layer_dims, class_weights)

epochs = 100
batch_size = 128
learning_rate = 0.01
patience = 15

train_loss_history, val_loss_history = [], []
train_acc_history, val_acc_history = [], []

best_val_loss = float('inf')
best_params = None
patience_counter = 0

print("\nStarting Training...")
for epoch in range(epochs):
    # Shuffle training data
    permutation = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[permutation]
    Y_train_shuffled = Y_train_oh[permutation]

    # Mini-batch SGD
    num_batches = int(np.ceil(X_train.shape[0] / batch_size))
    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min(start_idx + batch_size, X_train.shape[0])
        
        X_batch = X_train_shuffled[start_idx:end_idx]
        Y_batch = Y_train_shuffled[start_idx:end_idx]

        # Forward pass
        Y_hat_batch, caches = model.forward(X_batch)
        
        # Backward pass
        grads = model.backward(caches, Y_batch)
        
        # Update weights
        model.update_parameters(grads, learning_rate)

    # Evaluate full Epoch Metrics
    Y_hat_train, _ = model.forward(X_train)
    train_loss = model.compute_weighted_loss(Y_hat_train, Y_train_oh)
    train_acc = accuracy_score(y_train, np.argmax(Y_hat_train, axis=1))

    Y_hat_val, _ = model.forward(X_val)
    val_loss = model.compute_weighted_loss(Y_hat_val, Y_val_oh)
    val_acc = accuracy_score(y_val, np.argmax(Y_hat_val, axis=1))

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    print(f"Epoch {epoch+1:02d}/{epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = {k: v.copy() for k, v in model.params.items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"\nEarly stopping triggered at epoch {epoch+1}.")
        model.params = best_params
        break

# Restoring best model state
if best_params is not None:
    model.params = best_params

# =====================================================================
# 5. TEST EVALUATION & METRICS
# =====================================================================
Y_hat_test, _ = model.forward(X_test)
y_pred = np.argmax(Y_hat_test, axis=1)

overall_acc = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

print("\n" + "="*50)
print("TEST SET EVALUATION METRICS")
print("="*50)
print(f"Overall Accuracy: {overall_acc:.4f}")
print(f"Macro Precision:  {macro_p:.4f}")
print(f"Macro Recall:     {macro_r:.4f}")
print(f"Macro F1-Score:   {macro_f1:.4f}\n")

metrics_df = pd.DataFrame({
    'Class': [f"Class {i+1}" for i in range(num_classes)],
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
})
print(metrics_df.to_string(index=False))

# =====================================================================
# 6. VISUALIZATIONS & ANALYSIS
# =====================================================================
fig = plt.figure(figsize=(18, 12))

# Subplot 1: Training History (Loss & Accuracy)
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(train_loss_history, label='Train Loss', color='blue')
ax1.plot(val_loss_history, label='Val Loss', color='orange')
ax1.set_title('Loss Curves')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Weighted CE Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(train_acc_history, label='Train Accuracy', color='blue')
ax2.plot(val_acc_history, label='Val Accuracy', color='orange')
ax2.set_title('Accuracy Curves')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subplot 2: Class Imbalance Distribution
ax3 = fig.add_subplot(2, 3, 3)
ax3.bar(range(num_classes), class_counts, color='teal')
ax3.set_title('Dataset Class Distribution')
ax3.set_xlabel('Class (0-indexed)')
ax3.set_ylabel('Number of Samples')
ax3.set_xticks(range(num_classes))

# Subplot 3: Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
ax4 = fig.add_subplot(2, 3, 4)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4, cbar=False)
ax4.set_title('7x7 Confusion Matrix')
ax4.set_xlabel('Predicted Label')
ax4.set_ylabel('True Label')

# Subplot 4: Per-Class F1 Scores
ax5 = fig.add_subplot(2, 3, 5)
bars = ax5.bar(range(num_classes), f1, color='coral')
ax5.set_title('Per-Class F1-Scores')
ax5.set_xlabel('Class')
ax5.set_ylabel('F1-Score')
ax5.set_ylim(0, 1)
ax5.set_xticks(range(num_classes))
for bar in bars:
    yval = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontsize=8)

# Subplot 5: True vs Predicted Class Distribution Comparison
ax6 = fig.add_subplot(2, 3, 6)
pred_counts = np.bincount(y_pred, minlength=num_classes)
true_counts = np.bincount(y_test, minlength=num_classes)
x_indices = np.arange(num_classes)
width = 0.35

ax6.bar(x_indices - width/2, true_counts, width, label='True Test Dist', color='navy')
ax6.bar(x_indices + width/2, pred_counts, width, label='Predicted Dist', color='crimson')
ax6.set_title('True vs Predicted Class Distributions')
ax6.set_xlabel('Class')
ax6.set_ylabel('Count')
ax6.set_xticks(x_indices)
ax6.legend()