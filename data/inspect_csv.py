import pandas as pd

df = pd.read_csv("dataset-tickets-multi-lang-4-20k.csv")
print("COLUMNS:")
print(df.columns.tolist())
print()
print("FIRST 3 ROWS:")
print(df.head(3))