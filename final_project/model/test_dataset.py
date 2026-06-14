import pandas as pd

data = pd.read_csv(
    "../dataset/spam.csv",
    encoding="latin-1"
)

print(data.head())

print(data.columns.tolist())