
import os
import re

import torch
from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit import IndicProcessor

# Flask application
app = Flask(__name__, static_folder=".", static_url_path="")

# IndicTrans2 model setup
MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading IndicTrans2 model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
).to(DEVICE)

model.eval()

ip = IndicProcessor(inference=True)

print("Model loaded successfully!")
print("Device:", DEVICE)


# English-to-Hindi translation
def translate_to_hindi(sentence):
    src_lang = "eng_Latn"
    tgt_lang = "hin_Deva"

    batch = ip.preprocess_batch(
        [sentence],
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )

    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=False,
            max_length=256,
            num_beams=5,
            num_return_sequences=1
        )

    decoded = tokenizer.batch_decode(
        generated_tokens.detach().cpu(),
        skip_special_tokens=True
    )

    translations = ip.postprocess_batch(
        decoded,
        lang=tgt_lang
    )

    return translations[0]


# Serve the translator webpage
@app.route("/")
def home():
    return send_from_directory(".", "index.html")


# Translation API used by the webpage
@app.route("/api/translate", methods=["POST"])
def translate_api():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()

    if not text:
        return jsonify({
            "hindi_translation": "",
            "sentence_count": 0,
            "terms_detected": []
        }), 400

    if len(text) > 5000:
        return jsonify({
            "error": "Please enter fewer than 5000 characters."
        }), 413

    try:
        hindi = translate_to_hindi(text)

        sentence_count = len([
            sentence
            for sentence in re.split(r"[.!?]+", text)
            if sentence.strip()
        ])

        government_terms = [
            "government",
            "ministry",
            "citizen",
            "scheme",
            "policy",
            "certificate",
            "application",
            "department",
            "services",
            "official"
        ]

        terms_detected = [
            term for term in government_terms
            if term.lower() in text.lower()
        ]

        return jsonify({
            "hindi_translation": hindi,
            "sentence_count": sentence_count,
            "terms_detected": terms_detected
        })

    except Exception:
        app.logger.exception("Translation error")
        return jsonify({
            "error": "Translation failed. Please try again."
        }), 500


# Start the hosted application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
