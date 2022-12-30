import pandas as pd
from collections import Counter
import sys
import os
import numpy as np


def clean(context):
    temp = context.split("context :")[1].strip()
    temp = temp.replace('<|endoftext|>', ' ').strip()
    return temp


if __name__ == "__main__":
    base_path = sys.argv[1]
    sup_file = sys.argv[2]
    gen_file = sys.argv[3]

    train_df = pd.read_csv(sup_file)
    df_gen = pd.read_csv(gen_file)

    texts = []
    labels = []
    ids = []
    for i, row in df_gen.iterrows():
        context = row["generated_context"]
        clean_context = clean(context)
        if clean_context is None or len(clean_context) == 0:
            continue
        texts.append(clean_context)
        labels.append(row["label"])
        ids.append("gen_" + str(i))

    dic_df = pd.DataFrame.from_dict({"text": texts, "label": labels, "id": ids})
    print(Counter(dic_df["label"]))

    random_df = pd.concat([train_df, dic_df]).reset_index(drop=True)
    random_df = random_df.drop(columns=['id'])
    random_df["text"].replace("", np.nan, inplace=True)
    random_df = random_df.dropna(subset=["text"]).reset_index(drop=True)

    random_df.to_csv(os.path.join(base_path, "train_450_combined.csv"), index=False)
