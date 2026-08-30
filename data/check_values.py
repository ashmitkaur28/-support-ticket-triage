import pandas as pd

df = pd.read_csv("dataset-tickets-multi-lang-4-20k.csv")

print("QUEUE values:")
print(df["queue"].value_counts())
print()
print("PRIORITY values:")
print(df["priority"].value_counts())
print()
print("LANGUAGE values:")
print(df["language"].value_counts())
print()
print("TYPE values:")
print(df["type"].value_counts())