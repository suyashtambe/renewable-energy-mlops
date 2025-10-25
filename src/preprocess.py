# src/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess_data():
    # Load raw dataset
    df = pd.read_csv(r"C:\Users\Suyash Tambe\Desktop\renewable-energy-mlops\data\raw\global_energy_consumption.csv")

    # Check for missing values
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.mean())  # simple imputation

    # Keep 'Country' for per-country models
    # Target column
    target = "Renewable Energy Share (%)"
    if target not in df.columns:
        raise ValueError(f"Target column {target} not in dataset")

    # Temporal feature
    df["Year_diff"] = df["Year"] - df["Year"].min()

    # Optional: ratio features
    df["Industrial_to_Household_ratio"] = df["Industrial Energy Use (%)"] / \
                                          (df["Household Energy Use (%)"] + 1e-5)

    # Features to drop (keep numeric + Country)
    drop_cols = ["Year"]  # Year_diff replaces it
    df_processed = df.drop(columns=drop_cols)

    # Train/test split (stratify by Country if possible)
    train_df, test_df = train_test_split(
        df_processed, test_size=0.2, random_state=42, shuffle=True
    )

    # Create processed folder
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print(" Preprocessing complete!")
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

if __name__ == "__main__":
    preprocess_data()
