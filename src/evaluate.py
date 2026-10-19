import json
import spacy
import os

EVAL_PATH = "data/evaluation_examples.json"

if not os.path.exists(EVAL_PATH):
    print(f"File {EVAL_PATH} belum ditemukan. Jalankan src/prepare_eval.py terlebih dahulu.")
    exit()

with open(EVAL_PATH, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# Bandingkan baseline (en_core_web_sm) dengan model adaptasi (custom_ner_model)
models_to_evaluate = {
    "Baseline (en_core_web_sm)": spacy.load("en_core_web_sm"),
}

if os.path.exists("custom_ner_model"):
    models_to_evaluate["Adapted (custom_ner_model)"] = spacy.load("custom_ner_model")

for model_label, model_nlp in models_to_evaluate.items():
    print(f"\n==========================================")
    print(f"Evaluasi Model: {model_label}")
    print(f"==========================================")

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    errors = []

    for item in eval_data:
        text = f"{item['title']}\n\n{item['body']}"
        doc = model_nlp(text)

        # Standar pencocokan: exact-span and label matching
        pred_spans = {(ent.start_char, ent.end_char, ent.label_, ent.text) for ent in doc.ents}
        gold_spans = {(e['start'], e['end'], e['label'], e['text']) for e in item['entities']}

        tp = len(pred_spans.intersection(gold_spans))
        fp = len(pred_spans - gold_spans)
        fn = len(gold_spans - pred_spans)

        true_positives += tp
        false_positives += fp
        false_negatives += fn

        # Catat kasus galat untuk laporan analisis
        for item_fp in (pred_spans - gold_spans):
            errors.append(f"FP di '{item['id']}': model mendeteksi '{item_fp[3]}' sebagai {item_fp[2]}")
        for item_fn in (gold_spans - pred_spans):
            errors.append(f"FN di '{item['id']}': entitas '{item_fn[3]}' ({item_fn[2]}) terlewat")

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"True Positives  : {true_positives}")
    print(f"False Positives : {false_positives}")
    print(f"False Negatives : {false_negatives}")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"F1-Score        : {f1:.4f}")

    if errors:
        print("\nContoh Kasus Galat (3 Teratas untuk Laporan):")
        for err in errors[:3]:
            print(f" - {err}")