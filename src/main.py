import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
import spacy

app = FastAPI(title="Article NER Service")

# Muat model spaCy (gunakan en_core_web_sm sebagai base)
MODEL_NAME = "custom_ner_model" if os.path.exists("custom_ner_model") else "en_core_web_sm"
nlp = spacy.load(MODEL_NAME)

# Batas maksimum karakter yang diizinkan
MAX_INPUT_LENGTH = 15000


class ArticleRequest(BaseModel):
    title: str = Field(..., description="Article title")
    body: str = Field(..., description="Article body text")


# Tangani error parsing JSON bawaan agar tetap mengembalikan status 400
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return HTMLResponse(
        status_code=400,
        content='{"detail": "Malformed JSON or invalid data types provided."}',
        media_type="application/json"
    )


@app.get("/", response_class=HTMLResponse)
def serve_panel():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/extract-entities")
def extract_entities(payload: ArticleRequest):
    title = payload.title.strip()
    body = payload.body.strip()

    # Validasi: Tolak jika title atau body kosong
    if not title or not body:
        raise HTTPException(status_code=400, detail="Title and body must not be empty.")

    # Gabungkan teks judul dan isi
    combined_text = f"{title}\n\n{body}"

    # Validasi: Batas panjang teks
    if len(combined_text) > MAX_INPUT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Input length ({len(combined_text)}) exceeds maximum allowed ({MAX_INPUT_LENGTH})."
        )

    # Ekstraksi entitas dengan spaCy
    doc = nlp(combined_text)

    # Ambil entitas terdeteksi
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })

    return {
        "entities": entities,
        "meta": {
            "entity_count": len(entities),
            "model": MODEL_NAME
        }
    }