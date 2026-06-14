import pandas as pd

data = pd.read_csv(
    "../dataset/spam.csv",
    encoding="latin-1",
    header=None
)

print(data.head())
print("\nColumns:", data.columns.tolist())