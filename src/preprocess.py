# src/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess_data():
    # Define raw data path (relative to project root)
    raw_path = os.path.join("data", "raw", "global_energy_consumption.csv")

    # Ensure raw dataset exists
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f" Raw dataset not found at {raw_path}. Please ensure it exists before preprocessing.")

    # Load dataset
    df = pd.read_csv(raw_path)

    # Handle missing values
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.mean(numeric_only=True))  # safer for new pandas versions

    # Verify target column exists
    target = "Renewable Energy Share (%)"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in dataset. Found columns: {list(df.columns)}")

    # Create derived features
    df["Year_diff"] = df["Year"] - df["Year"].min()
    df["Industrial_to_Household_ratio"] = df["Industrial Energy Use (%)"] / (df["Household Energy Use (%)"] + 1e-5)

    # Drop columns not needed
    drop_cols = ["Year"]
    df_processed = df.drop(columns=drop_cols)

    # Train/test split
    train_df, test_df = train_test_split(df_processed, test_size=0.2, random_state=42, shuffle=True)

    # Save processed datasets
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print(" Preprocessing complete!")
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

if __name__ == "__main__":
    preprocess_data()
