import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function to load all review texts from a folder
def load_reviews(folder_path):
    reviews = []
    for fname in os.listdir(folder_path):
        with open(os.path.join(folder_path, fname), encoding='utf-8') as f:
            reviews.append(f.read())
    return reviews

# Load positive and negative reviews for training and testing
train_pos = load_reviews("acllmdb/train/pos")
train_neg = load_reviews("acllmdb/train/neg")
test_pos = load_reviews("acllmdb/test/pos")
test_neg = load_reviews("acllmdb/test/neg")

# Combine data and labels
X_train = train_pos + train_neg
y_train = [1] * len(train_pos) + [0] * len(train_neg)

X_test = test_pos + test_neg
y_test = [1] * len(test_pos) + [0] * len(test_neg)

# Vectorize text data using Bag-of-Words
vectorizer = CountVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_vectorized, y_train)

# Evaluate the model
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
