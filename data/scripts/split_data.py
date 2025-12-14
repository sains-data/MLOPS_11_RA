import pandas as pd
from sklearn.model_selection import train_test_split
import os

RANDOM_STATE = 42
TEST_SIZE = 0.2
PRODUCTION_SIZE = 0.1  # dari total data

# Load dataset
df = pd.read_csv("data/mushroom.csv")

# Split production data
df_temp, df_production = train_test_split(
    df,
    test_size=PRODUCTION_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["class"]
)

# Split train & test
df_train, df_test = train_test_split(
    df_temp,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df_temp["class"]
)

# Create directories
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/test", exist_ok=True)
os.makedirs("data/production", exist_ok=True)

# Save files
df_train.to_csv("data/train/mushroom_train.csv", index=False)
df_test.to_csv("data/test/mushroom_test.csv", index=False)
df_production.to_csv("data/production/mushroom_production.csv", index=False)

print("Data split completed:")
print(f"Train: {df_train.shape}")
print(f"Test: {df_test.shape}")
print(f"Production: {df_production.shape}")
