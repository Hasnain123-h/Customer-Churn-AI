from pathlib import Path
import pandas as pd
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR/"data"/"processed"/"cleaned_churn.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def create_features(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    
    y = y.map({"NO":0, "Yes":1})
    
    return X, y

if __name__ == "__main__":
    df = load_data()
    X, y = create_features(df)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    