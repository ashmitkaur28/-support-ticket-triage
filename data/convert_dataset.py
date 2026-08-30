import json
import random

import pandas as pd

random.seed(42)

df = pd.read_csv("dataset-tickets-multi-lang-4-20k.csv")

# Keep English only — the app's prompts are written in English
df = df[df["language"] == "en"].copy()

# Drop rows with missing subject/body/queue/priority — can't use those
df = df.dropna(subset=["subject", "body", "queue", "priority"])

# Combine subject + body into one ticket text
df["text"] = df["subject"].astype(str).str.strip() + "\n\n" + df["body"].astype(str).str.strip()

# Shuffle and take a manageable sample
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.head(200)  # 200 total: 50 for eval, 150 for demo

records = []
for i, row in df.iterrows():
    records.append({
        "id": f"kt{i:04d}",
        "text": row["text"],
        "true_category": row["queue"],
        "true_urgency": row["priority"],
    })

eval_records = records[:50]
demo_records = records[50:]

with open("../eval/eval_set.json", "w") as f:
    json.dump(eval_records, f, indent=2)

with open("tickets.json", "w") as f:
    json.dump(demo_records, f, indent=2)

print(f"Wrote {len(eval_records)} tickets to eval/eval_set.json")
print(f"Wrote {len(demo_records)} tickets to data/tickets.json")
print()
print("Category distribution in eval set:")
print(pd.Series([r["true_category"] for r in eval_records]).value_counts())