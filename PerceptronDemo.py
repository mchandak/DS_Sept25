import numpy as np

# ============================================================================
# STEP 1: PREPARE THE DATA
# ============================================================================

# Training data: [Credit Score, Income in thousands]
X = np.array([
    [0, 30],   # Bad Credit, $30K   → Reject
    [0, 80],   # Bad Credit, $80K   → Reject
    [1, 40],   # Good Credit, $40K  → Reject
    [1, 75]    # Good Credit, $75K  → Approve
])

# Expected outputs
y = np.array([0, 0, 0, 1])

# ============================================================================
# STEP 2: NORMALIZE THE DATA
# ============================================================================

# Normalize to 0-1 range (good practice when mixing different scales)
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_normalized = (X - X_min) / (X_max - X_min)
X = X_normalized

# ============================================================================
# STEP 3: INITIALIZE PARAMETERS
# ============================================================================

w1 = 0.0              # Weight for Credit Score
w2 = 0.0              # Weight for Income
b = 0.0               # Bias
learning_rate = 0.1
epochs = 20

# ============================================================================
# STEP 4: TRAINING LOOP
# ============================================================================

for epoch in range(epochs):
    for i in range(len(X)):
        credit = X[i][0]
        income = X[i][1]
        
        # Step 1: Calculate weighted sum
        z = w1 * credit + w2 * income + b
        
        # Step 2: Apply activation function
        y_pred = 1 if z >= 0 else 0
        
        # Step 3: Calculate error
        error = y[i] - y_pred
        
        # Step 4: Update weights and bias
        w1 = w1 + learning_rate * error * credit
        w2 = w2 + learning_rate * error * income
        b = b + learning_rate * error

# ============================================================================
# STEP 5: TESTING ON TRAINING DATA
# ============================================================================

for i in range(len(X)):
    credit = X[i][0]
    income = X[i][1]
    
    z = w1 * credit + w2 * income + b
    y_pred = 1 if z >= 0 else 0
    
    is_correct = "✓" if y_pred == y[i] else "✗"
    print(f"Sample {i+1}: z={z:.4f} → Pred={y_pred}, Actual={y[i]} {is_correct}")

# ============================================================================
# STEP 6: PREDICTIONS ON NEW CUSTOMERS
# ============================================================================

# New customers (need to normalize)
new_customers = [
    ([0, 25], "Bad Credit, $25K"),
    ([1, 85], "Good Credit, $85K"),
]

for original_input, description in new_customers:
    # Normalize the input
    normalized_input = (np.array(original_input) - X_min) / (X_max - X_min)
    
    credit_norm = normalized_input[0]
    income_norm = normalized_input[1]
    
    # Make prediction
    z = w1 * credit_norm + w2 * income_norm + b
    y_pred = 1 if z >= 0 else 0
    result = "APPROVE ✓" if y_pred == 1 else "REJECT ✗"
    
    print(f"\n{description}")
    print(f"  z = {z:.4f} → {result}")



