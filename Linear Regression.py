import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1️⃣ Load Dataset
df = pd.read_csv("your_dataset.csv")  # Replace with your actual dataset

# 2️⃣ Select Features and Target
X = df['feature_column'].values  # Independent variable
y = df['target_column'].values   # Dependent variable

# 3️⃣ Compute Means
x_mean = np.mean(X)
y_mean = np.mean(y)

# 4️⃣ Compute SSxy and SSxx
SSxy = np.sum((X - x_mean) * (y - y_mean))
SSxx = np.sum((X - x_mean) ** 2)

# 5️⃣ Compute Slope (m) and Intercept (b)
m = SSxy / SSxx
b = y_mean - m * x_mean

# 6️⃣ Define Prediction Function
def predict(x_val):
    return m * x_val + b

# 7️⃣ Make Predictions for Training Data
y_pred = np.array([predict(xi) for xi in X])

# 8️⃣ Calculate Mean Squared Error (MSE)
mse = np.mean((y - y_pred) ** 2)

# 9️⃣ Calculate R² Score
ss_total = np.sum((y - y_mean) ** 2)
ss_residual = np.sum((y - y_pred) ** 2)
r2_score = 1 - (ss_residual / ss_total)

# 🔟 Print Results
print(f"Slope (m): {m:.4f}")
print(f"Intercept (b): {b:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2_score:.4f}")

# 🔹 Plot Results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression from Scratch")
plt.legend()
plt.show()
