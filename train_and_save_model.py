import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# 👉 Load dataset
data = pd.read_csv(r'C:\Users\SysSoft\Documents\Final cor prjt\web creation\account_data.csv')  

# 👉 Preprocessing
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    if col != 'IsFraud':  
        data[col] = le.fit_transform(data[col].astype(str))

X = data.drop('IsFraud', axis=1)
y = data['IsFraud']

# 👉 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 👉 Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 👉 Train multiple models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

best_model = None
best_accuracy = 0
best_model_name = ""

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

print(f"\n✅ Best Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

# 👉 Save model + feature names
with open('best_model.pkl', 'wb') as model_file:
    pickle.dump({
        "model": best_model,
        "feature_names": X.columns.tolist()
    }, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("\n✅ best_model.pkl and scaler.pkl saved successfully!")
