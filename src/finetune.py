"""Fine-tune NLLB-200 (distilled 600M) for Kikuyu -> English.
Save as src/finetune.py and run from the project root:
    python src/finetune.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # use ONE GPU (avoids multi-GPU errors)

import json
import torch
import pandas as pd
import sacrebleu
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer,
                          DataCollatorForSeq2Seq)

SMOKE = False          # True = tiny test run, False = real run
MODEL = "facebook/nllb-200-distilled-600M"
SRC, TGT = "kik_Latn", "eng_Latn"
SPLITS = "Data/Processed/splits"
OUT = "models/nllb-kikuyu-en"
EPOCHS = 1 if SMOKE else 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

tok = AutoTokenizer.from_pretrained(MODEL, src_lang=SRC, tgt_lang=TGT)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)


def load(name):
    df = pd.read_csv(f"{SPLITS}/{name}.csv", encoding="utf-8-sig")
    df = df.dropna(subset=["Kikuyu", "English"])
    df["Kikuyu"] = df["Kikuyu"].astype(str).str.strip()
    df["English"] = df["English"].astype(str).str.strip()
    df = df[(df["Kikuyu"] != "") & (df["English"] != "")]
    return df.reset_index(drop=True)


train_df, val_df, test_df = load("train"), load("validation"), load("test")
if SMOKE:
    train_df, val_df, test_df = train_df.head(16), val_df.head(8), test_df.head(8)
print(f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")


def to_ds(df):
    return Dataset.from_pandas(df[["Kikuyu", "English"]])


def prep(batch):
    return tok(batch["Kikuyu"], text_target=batch["English"],
               truncation=True, max_length=128)


train_ds = to_ds(train_df).map(prep, batched=True, remove_columns=["Kikuyu", "English"])
val_ds = to_ds(val_df).map(prep, batched=True, remove_columns=["Kikuyu", "English"])


def translate(m, sentences, bs=16):
    m.to(device).eval()
    outputs = []
    for i in range(0, len(sentences), bs):
        batch = tok(sentences[i:i + bs], return_tensors="pt", padding=True,
                    truncation=True, max_length=128).to(device)
        with torch.no_grad():
            gen = m.generate(**batch,
                             forced_bos_token_id=tok.convert_tokens_to_ids(TGT),
                             max_new_tokens=128, num_beams=4)
        outputs += tok.batch_decode(gen, skip_special_tokens=True)
    return outputs


def score(m, df, label):
    preds = translate(m, df["Kikuyu"].tolist())
    refs = [df["English"].tolist()]
    bleu = sacrebleu.corpus_bleu(preds, refs).score
    chrf = sacrebleu.corpus_chrf(preds, refs).score
    print(f"{label}: BLEU={bleu:.2f}  chrF={chrf:.2f}")
    return preds, bleu, chrf


# 1) BEFORE fine-tuning (baseline)
base_preds, b_bleu, b_chrf = score(model, test_df, "BEFORE")

# 2) Fine-tune
args = Seq2SeqTrainingArguments(
    output_dir="checkpoints",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",            # no big checkpoints; we save the final model below
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    report_to="none",
)
trainer = Seq2SeqTrainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=DataCollatorForSeq2Seq(tok, model=model),
)
trainer.train()

# Save the model right away
os.makedirs(OUT, exist_ok=True)
trainer.model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print("Model saved to", OUT)

# 3) AFTER fine-tuning
ft_preds, a_bleu, a_chrf = score(trainer.model, test_df, "AFTER ")

# 4) Save evidence for your report
res = test_df[["Kikuyu", "English", "Category"]].copy()
res["Base_model"] = base_preds
res["Fine_tuned"] = ft_preds
res.to_csv("evaluation_results.csv", index=False, encoding="utf-8-sig")
with open("evaluation_scores.json", "w") as f:
    json.dump({"before": {"bleu": b_bleu, "chrf": b_chrf},
               "after": {"bleu": a_bleu, "chrf": a_chrf}}, f, indent=2)

# Per-category scores
for cat, g in res.groupby("Category"):
    s = sacrebleu.corpus_bleu(g["Fine_tuned"].tolist(), [g["English"].tolist()]).score
    print(f"  {cat}: BLEU={s:.2f}")