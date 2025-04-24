import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function to load data from directories
def load_data(data_dir):
    texts, labels = [], []
    for label in ['pos', 'neg']:
        label_dir = os.path.join(data_dir, label)
        for filename in os.listdir(label_dir):
            with open(os.path.join(label_dir, filename), encoding='utf-8') as f:
                texts.append(f.read())
            labels.append(1 if label == 'pos' else 0)
    return texts, labels

# Directories for training and testing data
train_dir = "datasets/aclImdb/train/"
test_dir = "datasets/aclImdb/test/"

# Load train and test data
X_train, y_train = load_data(train_dir)
X_test, y_test = load_data(test_dir)

# Vectorize text data
vectorizer = CountVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_vectorized, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test_vectorized)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
