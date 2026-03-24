import torch
import os
import pandas as pd

from torch.optim import AdamW
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from DNA_BERT_encoder import DNAEncoder
from transformers import DataCollatorForLanguageModeling

# >>>>>>>>>>> UTILITIES <<<<<<<<<<<<
def BIOSCAN_preprocessing(sequence):
    sequence = sequence.upper()
    sequence = sequence.replace("N", "")
    sequence = sequence.replace("-", "")

    return sequence.strip()

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        trust_remote_code=True
    )
    model = AutoModelForMaskedLM.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        trust_remote_code=True
    )
    model.cuda()
    model.train()

    optimizer = AdamW(model.parameters(), lr=2e-5)

    return tokenizer, optimizer, model

def get_collator(tokenizer):
    """Makes masked tokens for fine-tuning."""
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

def save_checkpoint(model, optimizer, step):
    os.makedirs("checkpoints", exist_ok=True)

    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, f"checkpoints/dna_bert_step_{step}.pt")

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["step"]

# >>>>>>>>>>> Tokenization <<<<<<<<<<<<
def sequence_stream(file="./csv/BIOSCAN_5M_Insect_Dataset_metadata.csv", chunk_size=5000):
    for chunk in pd.read_csv(
        file,
        usecols=["dna_barcode"],
        chunksize=chunk_size
    ):
        sequences = chunk["dna_barcode"].dropna().tolist()
        yield sequences

def tokenize_batch(tokenizer, sequences):
    return tokenizer(
        sequences,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

# >>>>>>>>>>> MAIN TRAINING LOOP <<<<<<<<<<<<
def training_loop(model, tokenizer, optimizer, chunk_size=5000):
    collator = get_collator(tokenizer)
    step = 0

    for sequences in sequence_stream(chunk_size=chunk_size):
        sequences = [BIOSCAN_preprocessing(s) for s in sequences if len(s) > 100]
        tokens = tokenize_batch(tokenizer, sequences)
        batch = collator(tokens)

        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        labels = batch["labels"].cuda()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        if step % 500 == 0:
            print(f"step {step} loss {loss.item():.4f}")
            save_checkpoint(model, optimizer, step)


if __name__ == "__main__":
    chunk_size = 10000
    tokenizer, optimizer, model = load_model()
    training_loop(model, tokenizer, optimizer, chunk_size)
