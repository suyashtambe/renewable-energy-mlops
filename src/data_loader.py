import pandas as pd
import yaml

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def load_data():
    config = load_config()
    data = pd.read_csv(config["data"]["raw_path"])
    return data

if __name__ == "__main__":
    data = load_data()
    print(data.head())
