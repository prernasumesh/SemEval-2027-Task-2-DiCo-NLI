import os
import pandas as pd
df = pd.read_csv("final_data/dev/dico_nli_dev_track1_submission_template.csv")
df["label"] = "FORWARD_ENTAILMENT"
os.makedirs("results", exist_ok=True)
df[["instance_id", "label"]].to_csv("results/majority_dev.csv", index=False)
print(df.shape)
print(df.head())