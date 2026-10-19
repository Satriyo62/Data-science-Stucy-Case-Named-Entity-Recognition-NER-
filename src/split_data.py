import pandas as pd
import glob
import numpy as np

# Muat seluruh data CSV
csv_files = glob.glob("D:\\uts_data science\\*.csv")
df = pd.concat([pd.read_csv(f, sep=';', quoting=0) for f in csv_files], ignore_index=True)

# Bersihkan spasi/tanda kutip pada kolom language dan filter hanya artikel bahasa Inggris
df['language'] = df['language'].astype(str).str.strip("'\" ")
df_en = df[df['language'] == 'en'].reset_index(drop=True)

# Dapatkan ID unik artikel
unique_ids = df_en['original_id'].dropna().unique()

# Deterministic seed sesuai ketentuan studi kasus
np.random.seed(42)
np.random.shuffle(unique_ids)

# Hitung split 70:15:15
n_total = len(unique_ids)
n_train = int(0.70 * n_total)
n_val = int(0.15 * n_total)

train_ids = set(unique_ids[:n_train])
val_ids = set(unique_ids[n_train:n_train + n_val])
test_ids = set(unique_ids[n_train + n_val:])

train_df = df_en[df_en['original_id'].isin(train_ids)]
val_df = df_en[df_en['original_id'].isin(val_ids)]
test_df = df_en[df_en['original_id'].isin(test_ids)]

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print(f"Total artikel bahasa Inggris: {n_total}")
print(f"Train split : {len(train_df)} artikel")
print(f"Val split   : {len(val_df)} artikel")
print(f"Test split  : {len(test_df)} artikel")