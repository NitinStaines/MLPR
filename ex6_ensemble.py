import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Load and prepare data
data = pd.read_csv("bill_authentication.csv")
X = data.drop(columns=["Class"]).values
y = data["Class"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Unified evaluation and plotting function
def evaluate_model(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Fake", "Genuine"], yticklabels=["Fake", "Genuine"])
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Bagging: Random Forest
evaluate_model("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42))

# Boosting: XGBoost
evaluate_model("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))

# Stacking: KNN + Decision Tree → SVM
base_models = [
    ('knn', KNeighborsClassifier(n_neighbors=3)),
    ('dt', DecisionTreeClassifier(max_depth=3))
]
stacking = StackingClassifier(estimators=base_models, final_estimator=SVC(kernel='linear'))
evaluate_model("Stacking", stacking)
