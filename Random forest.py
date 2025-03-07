import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ Load Dataset (Replace with your dataset)
df = pd.read_csv("your_dataset.csv")  # Change the filename as needed

# 2️⃣ Handle Missing Values (Fill NaN with mode for categorical, median for numerical)
for col in df.columns:
    if df[col].dtype == 'object':  # Categorical column
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:  # Numerical column
        df[col].fillna(df[col].median(), inplace=True)

# 3️⃣ Encode Categorical Columns (Convert text to numbers)
encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

# 4️⃣ Define Features (X) and Target (y)
X = df.drop(columns=['target'])  # Replace 'target' with actual target column name
y = df['target']

# 5️⃣ Split Data into Training and Testing Sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Train Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=10,      # Maximum tree depth (tune as needed)
    random_state=42
)
rf_model.fit(X_train, y_train)  # Train the model

# 7️⃣ Make Predictions
y_pred = rf_model.predict(X_test)

# 8️⃣ Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("Classification Report:\n", classification_report(y_test, y_pred))
