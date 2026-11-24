import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE = "facebook/nllb-200-distilled-600M"
FINETUNED = "models/nllb-kikuyu-en"
SRC, TGT = "kik_Latn", "eng_Latn"
device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.isdir(FINETUNED):
    raise SystemExit(f"Model folder '{FINETUNED}' not found. "
                     "Run from the project root.")

tok = AutoTokenizer.from_pretrained(FINETUNED, src_lang=SRC, tgt_lang=TGT)
ft_model = AutoModelForSeq2SeqLM.from_pretrained(FINETUNED).to(device).eval()
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE).to(device).eval()


def translate(model, text):
    batch = tok(text, return_tensors="pt", truncation=True,
                max_length=128).to(device)
    with torch.no_grad():
        out = model.generate(**batch,
                             forced_bos_token_id=tok.convert_tokens_to_ids(TGT),
                             max_new_tokens=128, num_beams=4)
    return tok.batch_decode(out, skip_special_tokens=True)[0]


def run(text):
    text = (text or "").strip()
    if not text:
        return "", ""
    return translate(ft_model, text), translate(base_model, text)


examples = [
    ["Ihinda rīega rīa kūhanda mīembe nī mweri wa gatatū, wa kana na wa gatano rīrīa gūtarī mbura nene kana riūa rīingī."],
    ["Ririkana kūhuhira ihinda rīrīa rīagīrīire mīembe īkīambīrīria kūruta kīro."],
]

demo = gr.Interface(
    fn=run,
    inputs=gr.Textbox(lines=4, label="Kikuyu text (e.g. a farmer's report)"),
    outputs=[gr.Textbox(label="English (fine-tuned model)"),
             gr.Textbox(label="English (base NLLB, for comparison)")],
    title="Kikuyu → English Translator for Farmers",
    description="Fine-tuned NLLB-200 so Kikuyu-speaking farmers can report "
                "crop problems to English-speaking officers.",
    examples=examples,
)

if __name__ == "__main__":
    demo.launch(share=True)