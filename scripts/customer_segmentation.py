"""
customer_segmentation.py

1. Loads insurance customer data.
2. Extracts RFM-like features (Recency, Frequency, Monetary) + optional extras.
3. Applies K-Means clustering to segment customers.
4. Saves results to 'customer_segments.csv'.
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join("..", "data", "insurance_customers.csv")
OUTPUT_PATH = os.path.join("..", "data", "customer_segments.csv")

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    
    # Basic cleanup (if needed)
    df.fillna(0, inplace=True)  # simplistic approach for missing data
    
    # Example: We'll pick these columns for clustering:
    # Recency, Frequency, Monetary are typical for RFM
    # plus an optional "Num_Claims" or "Policy_Premium" if relevant.
    # Adjust as needed for your data.
    features = ["Recency", "Frequency", "Monetary"]
    # Ensure columns exist in the dataset
    for col in features:
        if col not in df.columns:
            print(f"Column '{col}' not found in dataset. Skipping.")
            return

    # Subset the data
    X = df[features].copy()
    
    # Optional: Standardize features to give them equal weight
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means Clustering
    # Choose a cluster count (k=4 as an example, can tune with elbow method)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
    df["Segment"] = kmeans.fit_predict(X_scaled)
    
    # Optionally compute cluster centers or other stats
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(cluster_centers, columns=features)
    print("Cluster Centers (unscaled):\n", centers_df)
    
    # Save segmentation results
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Customer segments saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
