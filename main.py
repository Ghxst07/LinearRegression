# ## IMPORT STATEMENTS
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ## Opening the CSV file as dataframe
# %%
df = pd.read_csv('coffee_shop_revenue.csv')
print(df.head())
print("Shape of the dataset:", df.shape)


# ## Setting up Y = W.X + B
# %%
def function(w, x, b):
    return np.dot(x, w) + b


# ## Setting up Cost Function
# %%
def cost_function(w, x, y, b):
    m = x.shape[0]
    f = function(w, x, b)
    squ_sum = np.sum((f - y) ** 2)
    cost = squ_sum / (2 * m)
    return cost


# ## Finding Partial Derivative
# %%
def partial_derivative(w, x, y, b):
    m = x.shape[0]
    f = function(w, x, b)
    dw = np.dot(x.T, (f - y)) / m  # Partial derivative w.r.t. w
    db = np.sum(f - y) / m  # Partial derivative w.r.t. b
    return dw, db


# ## Setting up Gradient Descent
# %%
def gradient_descent(w, x, y, b, alpha, iterations):
    cost_history = []
    for i in range(iterations):
        dw, db = partial_derivative(w, x, y, b)
        w = w - alpha * dw  # Update weights
        b = b - alpha * db  # Update bias
        cost = cost_function(w, x, y, b)
        cost_history.append(cost)
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost}")
    return w, b, cost_history


# %%
def main():
    # Extract features (X) and target (Y)
    x = df[df.columns[1]].to_numpy().reshape(-1, 1)  # Single feature
    y = df[df.columns[-1]].to_numpy().reshape(-1, 1)  # Target

    # Normalize x if needed
    x_norm = (x - np.mean(x)) / np.std(x)

    w = np.random.normal(loc=0.0, scale=0.01, size=(x.shape[1], 1))  # Random small values for w
    b = np.zeros((1, 1))

    # Hyperparameters
    iterations = 1000
    alpha = 0.004

    # Train using x_norm
    w, b, cost_history = gradient_descent(w, x_norm, y, b, alpha, iterations)

    # Predict using x_norm
    y_pred = function(w, x_norm, b)

    # Now plot against the original x (undo normalization for x-axis if needed)
    plt.scatter(x, y, color="blue", label="Actual Data")
    plt.plot(x, y_pred, color="red", label="Regression Line")

    plt.legend()
    plt.show()


# Run the main function
if __name__ == '__main__':
    main()

