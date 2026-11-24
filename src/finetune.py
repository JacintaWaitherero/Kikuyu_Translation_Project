"""Bidirectional fine-tuning of NLLB-200 (distilled 600M): Kikuyu <-> English.
Save as src/finetune.py and run from the project root:
    python src/finetune.py

Optional: put corrected word/phrase pairs in Data/glossary.csv
(columns: English,Kikuyu). They are added to training several times
so the model learns the correct translation.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # use ONE GPU

import json
import random
import torch
import pandas as pd
import sacrebleu
from datasets import Dataset, concatenate_datasets
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer,
                          DataCollatorForSeq2Seq)

SMOKE = False            # True = tiny test run, False = real run
MODEL = "facebook/nllb-200-distilled-600M"
KIK, ENG = "kik_Latn", "eng_Latn"
SPLITS = "Data/Processed/splits"
GLOSSARY = "Data/glossary.csv"      # optional
GLOSSARY_REPEAT = 10                # how many times glossary rows are repeated
OUT = "models/nllb-kikuyu-en"
EPOCHS = 1 if SMOKE else 5
SEED = 42
random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# One tokenizer per direction (sets the correct language tags)
tok_k2e = AutoTokenizer.from_pretrained(MODEL, src_lang=KIK, tgt_lang=ENG)
tok_e2k = AutoTokenizer.from_pretrained(MODEL, src_lang=ENG, tgt_lang=KIK)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)


def load(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.dropna(subset=["Kikuyu", "English"])
    df["Kikuyu"] = df["Kikuyu"].astype(str).str.strip()
    df["English"] = df["English"].astype(str).str.strip()
    df = df[(df["Kikuyu"] != "") & (df["English"] != "")]
    return df.reset_index(drop=True)


train_df = load(f"{SPLITS}/train.csv")
val_df = load(f"{SPLITS}/validation.csv")
test_df = load(f"{SPLITS}/test.csv")

# Glossary: corrected words/phrases, repeated so the model pays attention
if os.path.exists(GLOSSARY):
    gl = load(GLOSSARY)
    gl["Category"] = "Glossary"
    train_df = pd.concat([train_df] + [gl] * GLOSSARY_REPEAT, ignore_index=True)
    print(f"Glossary: {len(gl)} pairs x {GLOSSARY_REPEAT}")

if SMOKE:
    train_df, val_df, test_df = train_df.head(16), val_df.head(8), test_df.head(8)
print(f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")


def make_ds(df, src_col, tok):
    tgt_col = "English" if src_col == "Kikuyu" else "Kikuyu"
    ds = Dataset.from_pandas(df[[src_col, tgt_col]].rename(
        columns={src_col: "src", tgt_col: "tgt"}))
    return ds.map(
        lambda b: tok(b["src"], text_target=b["tgt"],
                      truncation=True, max_length=128),
        batched=True, remove_columns=["src", "tgt"])


def both_directions(df):
    k2e = make_ds(df, "Kikuyu", tok_k2e)
    e2k = make_ds(df, "English", tok_e2k)
    return concatenate_datasets([k2e, e2k]).shuffle(seed=SEED)


train_ds = both_directions(train_df)
val_ds = both_directions(val_df)


def translate(m, sentences, direction, bs=16):
    tok = tok_k2e if direction == "kik2eng" else tok_e2k
    tgt = ENG if direction == "kik2eng" else KIK
    m.to(device).eval()
    outputs = []
    for i in range(0, len(sentences), bs):
        batch = tok(sentences[i:i + bs], return_tensors="pt", padding=True,
                    truncation=True, max_length=128).to(device)
        with torch.no_grad():
            gen = m.generate(**batch,
                             forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
                             max_new_tokens=128, num_beams=4)
        outputs += tok.batch_decode(gen, skip_special_tokens=True)
    return outputs


def score_both(m, label):
    """Score both directions on the test set. Returns predictions + scores."""
    results = {}
    for direction, src_col, tgt_col in [("kik2eng", "Kikuyu", "English"),
                                        ("eng2kik", "English", "Kikuyu")]:
        preds = translate(m, test_df[src_col].tolist(), direction)
        refs = [test_df[tgt_col].tolist()]
        bleu = sacrebleu.corpus_bleu(preds, refs).score
        chrf = sacrebleu.corpus_chrf(preds, refs).score
        print(f"{label} {direction}: BLEU={bleu:.2f}  chrF={chrf:.2f}")
        results[direction] = {"preds": preds, "bleu": bleu, "chrf": chrf}
    return results


# 1) BEFORE fine-tuning
before = score_both(model, "BEFORE")

# 2) Fine-tune (both directions mixed together)
args = Seq2SeqTrainingArguments(
    output_dir="checkpoints",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    fp16=torch.cuda.is_available(),
    logging_steps=100,
    report_to="none",
)
trainer = Seq2SeqTrainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=DataCollatorForSeq2Seq(tok_k2e, model=model),
)
trainer.train()

os.makedirs(OUT, exist_ok=True)
trainer.model.save_pretrained(OUT)
tok_k2e.save_pretrained(OUT)
print("Model saved to", OUT)

# 3) AFTER fine-tuning
after = score_both(trainer.model, "AFTER ")

# 4) Save evidence
rows = []
for direction, src_col, tgt_col in [("kik2eng", "Kikuyu", "English"),
                                    ("eng2kik", "English", "Kikuyu")]:
    part = test_df[[src_col, tgt_col, "Category"]].copy()
    part.columns = ["Source", "Reference", "Category"]
    part["Direction"] = direction
    part["Base_model"] = before[direction]["preds"]
    part["Fine_tuned"] = after[direction]["preds"]
    rows.append(part)
res = pd.concat(rows, ignore_index=True)
res.to_csv("evaluation_results.csv", index=False, encoding="utf-8-sig")

with open("evaluation_scores.json", "w") as f:
    json.dump({d: {"before": {"bleu": before[d]["bleu"], "chrf": before[d]["chrf"]},
                   "after": {"bleu": after[d]["bleu"], "chrf": after[d]["chrf"]}}
               for d in ["kik2eng", "eng2kik"]}, f, indent=2)

# Per-category scores
for direction in ["kik2eng", "eng2kik"]:
    print(f"\nPer-category BLEU ({direction}):")
    sub = res[res["Direction"] == direction]
    for cat, g in sub.groupby("Category"):
        s = sacrebleu.corpus_bleu(g["Fine_tuned"].tolist(),
                                  [g["Reference"].tolist()]).score
        print(f"  {cat}: BLEU={s:.2f}")