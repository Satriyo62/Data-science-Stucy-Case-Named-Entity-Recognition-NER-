import spacy
from spacy.training.example import Example
import random
import os

# Muat model dasar
nlp = spacy.load("en_core_web_sm")

# Siapkan data latih spesifik domain berita (contoh representatif dari train split)
TRAIN_DATA = [
    (
        "Microsoft and OpenAI announced a partnership in London yesterday.",
        {"entities": [(0, 9, "ORG"), (14, 20, "ORG"), (48, 54, "GPE"), (55, 64, "DATE")]}
    ),
    (
        "Tim Cook visited the Apple headquarters in California.",
        {"entities": [(0, 8, "PERSON"), (21, 26, "ORG"), (43, 53, "GPE")]}
    ),
    (
        "Reuters reported that the United Nations held a summit in New York on 2026-03-15.",
        {"entities": [(0, 7, "ORG"), (26, 40, "ORG"), (58, 66, "GPE"), (70, 80, "DATE")]}
    ),
    (
        "Satya Nadella discussed cloud growth during the conference.",
        {"entities": [(0, 13, "PERSON")]}
    ),
    (
        "Google and Alphabet expanded their research facility in Zurich.",
        {"entities": [(0, 6, "ORG"), (11, 19, "ORG"), (56, 62, "GPE")]}
    )
]

# Ambil komponen NER dari pipeline
ner = nlp.get_pipe("ner")

# Daftarkan label tambahan jika belum ada
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Nonaktifkan komponen lain selama pelatihan bobot NER
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

print("Memulai adaptasi bobot model NER...")
random.seed(42)
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.resume_training()
    for iteration in range(20):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.2, sgd=optimizer, losses=losses)
        print(f"Iterasi {iteration + 1}/20 - Loss: {losses.get('ner', 0.0):.4f}")

# Simpan model hasil adaptasi
OUTPUT_DIR = "custom_ner_model"
nlp.to_disk(OUTPUT_DIR)
print(f"\nModel hasil adaptasi berhasil disimpan di folder: {OUTPUT_DIR}")