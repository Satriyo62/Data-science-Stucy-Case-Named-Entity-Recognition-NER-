import pandas as pd
import glob
import os

# Sesuaikan path ke folder sampel data kamu
DATA_PATH = "D:\\uts_data science\\*.csv" 
csv_files = glob.glob(DATA_PATH)

if not csv_files:
    print(f"File CSV tidak ditemukan di: {DATA_PATH}")
    exit()

print(f"Ditemukan {len(csv_files)} file CSV.")

# Muat dan gabungkan data dengan pemisah titik koma
dfs = [pd.read_csv(f, sep=';', quoting=0) for f in csv_files]
df = pd.concat(dfs, ignore_index=True)

# 1. Total baris
print(f"Total baris dimuat: {len(df)}")

# 2. Cek baris duplikat
duplicate_rows = df.duplicated().sum()
print(f"Jumlah baris duplikat: {duplicate_rows}")

# 3. Cek data kosong pada title dan body
missing_title = df['title'].isna().sum()
missing_body = df['body'].isna().sum()
print(f"Missing title: {missing_title}")
print(f"Missing body: {missing_body}")

# 4. Distribusi bahasa
print("\nDistribusi Bahasa:")
print(df['language'].value_counts(dropna=False))

# 5. Sampel kolom entities
print("\nContoh 3 isi kolom entities:")
print(df['entities'].dropna().head(3).tolist())