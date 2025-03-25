"""
model_training.py

Trains three models (Linear, RF, GB) without dropping any dummy categories.
All categories remain, so we have consistent columns at inference time.
"""

import os
import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = os.path.join("..", "data", "insurance_customers.csv")
df = pd.read_csv(DATA_PATH)

# Preprocess
numeric_cols = ["Age", "Policy_Premium", "Num_Claims", "Frequency", 
                "Recency", "Monetary", "Historical_CLV"]
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

df.dropna(subset=["Historical_CLV"], inplace=True)

# IMPORTANT: do NOT drop_first
categorical_cols = ["Gender", "Province", "Customer_Feedback"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

X = df.drop(["CustomerID", "Historical_CLV"], axis=1)
y = df["Historical_CLV"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

metrics_data = []

for model_name, model_obj in models.items():
    model_obj.fit(X_train, y_train)
    y_pred = model_obj.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{model_name}: RMSE={rmse:.2f}, R^2={r2:.3f}")
    
    # Save model
    models_path = os.path.join("..", "models")
    os.makedirs(models_path, exist_ok=True)
    model_filename = os.path.join(models_path, f"model_{model_name.lower()}.pkl")
    with open(model_filename, "wb") as f:
        pickle.dump(model_obj, f)
    
    metrics_data.append({"model": model_name, "RMSE": rmse, "R2": r2})

metrics_df = pd.DataFrame(metrics_data)
metrics_csv_path = os.path.join("..", "models", "model_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)

print("All models trained and saved.")
print(f"Metrics saved to {metrics_csv_path}")
