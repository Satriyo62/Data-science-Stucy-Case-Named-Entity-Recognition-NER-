import pandas as pd
import spacy
import json

nlp = spacy.load("en_core_web_sm")
df_test = pd.read_csv("data/test.csv")

# Ambil 9 artikel dengan panjang wajar untuk evaluasi
samples = df_test.dropna(subset=['title', 'body']).head(9)

eval_data = []

for idx, row in samples.iterrows():
    title = str(row['title']).strip()
    # Ambil 1-2 kalimat pertama body agar ringkas dan mudah diverifikasi
    body = str(row['body']).strip()[:350]
    
    combined_text = f"{title}\n\n{body}"
    doc = nlp(combined_text)
    
    # Ambil entitas yang didukung: PERSON, ORG, GPE, LOC, DATE
    supported_labels = {"PERSON", "ORG", "GPE", "LOC", "DATE", "EVENT"}
    entities = []
    for ent in doc.ents:
        if ent.label_ in supported_labels:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
            
    eval_data.append({
        "id": f"test-eval-{len(eval_data)+1:02d}",
        "original_id": str(row['original_id']),
        "title": title,
        "body": body,
        "entities": entities
    })

# Tambahkan 1 sampel artikel tanpa entitas sama sekali (syarat wajib studi kasus)
eval_data.append({
    "id": "test-eval-10",
    "original_id": "manual-clean-01",
    "title": "Healthy Daily Routines for Productivity",
    "body": "Drinking water regularly and getting enough sleep every night improves concentration and general wellbeing.",
    "entities": []
})

with open("data/evaluation_examples.json", "w", encoding="utf-8") as f:
    json.dump(eval_data, f, indent=2, ensure_ascii=False)

print(f"Berhasil membuat data/evaluation_examples.json dengan {len(eval_data)} artikel!")