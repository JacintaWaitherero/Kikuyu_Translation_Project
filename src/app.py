"""Gradio demo: Kikuyu <-> English translation with the fine-tuned NLLB model.
Save as src/app.py and run from the project root:
    pip install gradio
    python src/app.py
"""
import os
import re
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE = "facebook/nllb-200-distilled-600M"
FINETUNED = "models/nllb-kikuyu-en"
KIK, ENG = "kik_Latn", "eng_Latn"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Quick safety net for words that are STILL wrong after retraining.
# Format: "wrong output": "correct output"  (applied to the model's output)
FIXES = {
    # "wrong word": "right word",
}

if not os.path.isdir(FINETUNED):
    raise SystemExit(f"Model folder '{FINETUNED}' not found. "
                     "Run from the project root.")

tok_k2e = AutoTokenizer.from_pretrained(FINETUNED, src_lang=KIK, tgt_lang=ENG)
tok_e2k = AutoTokenizer.from_pretrained(FINETUNED, src_lang=ENG, tgt_lang=KIK)
ft_model = AutoModelForSeq2SeqLM.from_pretrained(FINETUNED).to(device).eval()
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE).to(device).eval()

DIRECTIONS = {"Kikuyu → English": "kik2eng", "English → Kikuyu": "eng2kik"}


def apply_fixes(text):
    for wrong, right in FIXES.items():
        text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
    return text


def translate(model, text, direction):
    tok = tok_k2e if direction == "kik2eng" else tok_e2k
    tgt = ENG if direction == "kik2eng" else KIK
    batch = tok(text, return_tensors="pt", truncation=True,
                max_length=128).to(device)
    with torch.no_grad():
        out = model.generate(**batch,
                             forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
                             max_new_tokens=128, num_beams=4)
    return tok.batch_decode(out, skip_special_tokens=True)[0]


def run(text, direction_label):
    text = (text or "").strip()
    if not text:
        return "", ""
    d = DIRECTIONS[direction_label]
    return (apply_fixes(translate(ft_model, text, d)),
            translate(base_model, text, d))


examples = [
    ["Ihinda rīega rīa kūhanda mīembe nī mweri wa gatatū, wa kana na wa gatano rīrīa gūtarī mbura nene kana riūa rīingī.", "Kikuyu → English"],
    ["The best practice is to water until the plantlet starts developing new leaves.", "English → Kikuyu"],
]

demo = gr.Interface(
    fn=run,
    inputs=[gr.Textbox(lines=4, label="Type or paste text"),
            gr.Radio(list(DIRECTIONS), value="Kikuyu → English",
                     label="Direction")],
    outputs=[gr.Textbox(label="Translation (fine-tuned model)"),
             gr.Textbox(label="Translation (base NLLB, for comparison)")],
    title="Kikuyu ⇄ English Translator for Farmers",
    description="A Kikuyu farmer reports a crop problem in Kikuyu and the "
                "officer reads it in English. The officer's English advice "
                "goes back to the farmer in Kikuyu.",
    examples=examples,
)

if __name__ == "__main__":
    demo.launch(share=True)