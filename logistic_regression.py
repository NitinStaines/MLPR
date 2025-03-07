class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = []
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + self.exp(-z))

    def exp(self, x):
        e = 2.71828
        return e ** x

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        self.weights = [0] * n_features
        self.bias = 0

        for _ in range(self.iterations):
            for i in range(n_samples):
                linear_model = self.bias
                for j in range(n_features):
                    linear_model += self.weights[j] * X[i][j]

                prediction = self.sigmoid(linear_model)

                error = prediction - y[i]

                # Updating Weights
                for j in range(n_features):
                    self.weights[j] -= self.learning_rate * error * X[i][j]

                # Updating Bias
                self.bias -= self.learning_rate * error

    def predict(self, X):
        predictions = []
        for sample in X:
            linear_model = self.bias
            for j in range(len(sample)):
                linear_model += self.weights[j] * sample[j]
            prediction = self.sigmoid(linear_model)
            predictions.append(1 if prediction > 0.5 else 0)
        return predictions


# Dataset with Non-Numerical Features
Gender = ['Male', 'Female', 'Male', 'Female']
Experience = [2, 3, 5, 1]
Salary = [50000, 60000, 80000, 40000]
Target = [1, 0, 1, 0]

# Label Encoding for Gender
Gender_encoded = [1 if g == 'Male' else 0 for g in Gender]

# Prepare Features
X = [[Gender_encoded[i], Experience[i], Salary[i]] for i in range(len(Gender))]
y = Target

# Train Model
model = LogisticRegression(learning_rate=0.001, iterations=1000)
model.fit(X, y)

print("\n✅ Model Trained Successfully")
print("Weights:", model.weights)
print("Bias:", model.bias)

# Prediction
features = list(map(float, input("\nEnter Gender (1 for Male, 0 for Female), Experience, Salary: ").split()))
predicted = model.predict([features])
print("\nPredicted Class:", predicted[0])
