import numpy as np

# Sample data: y = 2x
X = np.array([1,2,3,4,5])
y = np.array([2,4,6,8,10])

# Initialize weight
w = 0.0
lr = 0.1
batch_size = 2

# Mini-batch Gradient Descent
for epoch in range(5):
    # Shuffle data
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Process mini-batches
    for i in range(0, len(X), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        
        # Predictions
        y_pred = w * X_batch
        
        # Compute gradient
        grad = -2 * np.sum(X_batch * (y_batch - y_pred)) / batch_size
        
        # Update weight
        w = w - lr * grad
    print(f"Epoch {epoch+1}, Weight: {w:.4f}")