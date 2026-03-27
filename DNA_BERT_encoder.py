import torch
import os
import pandas as pd

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import AutoModel
from transformers import DataCollatorForLanguageModeling

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets import load_dataset
from functools import partial

class DNAEncoder(nn.Module):
    def __init__(self, model_name, num_classes, lr=2e-5):
        super().__init__()

        self.bert = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )

        self.class_head = nn.Linear(512, num_classes)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:, 0, :]
        embedding = self.projection(cls)

        logits = self.class_head(embedding)
        return logits
    
    def fit(self, dataloader, eval_dataloader, epochs, device):
        metrics = {"train_loss": [], "eval_loss": [], "accuracy": []}
        for epoch in range(epochs):
            self.train()
            train_loss = 0

            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                self.optimizer.zero_grad()

                logits = self.forward(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(dataloader)
            metrics["train_loss"].append(avg_train_loss)

            # Evaluation
            self.eval()
            eval_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in eval_dataloader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    logits = self.forward(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)

                    eval_loss += loss.item()

                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            # Calculate Metrics
            avg_eval_loss = eval_loss / len(eval_dataloader)
            accuracy = correct / total
            metrics["eval_loss"].append(avg_eval_loss)
            metrics["accuracy"].append(accuracy)
            # Print Metrics
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_eval_loss:.4f}, Acc: {accuracy:.4f}")

        return metrics
        
    
def collate(batch, tokenizer, species_dict, max_length=512):
    sequences = [item["dna_barcode"] for item in batch]
    labels = [species_dict[item["species"]] for item in batch]

    tokens = tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long)
    }

def training_loop():
    model_handle = 'zhihan1996/DNA_bert_3'

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_handle, trust_remote_code=True)

    # Load the BIOSCAN5M train dataset
    train_dataset = load_dataset("dataset.py", name="cropped_256_train", split="train", trust_remote_code=True)
    train_dataset = train_dataset.with_format("torch", device=device)

    # Load the BIOSCAN5M validation dataset
    eval_dataset = load_dataset("dataset.py", name="cropped_256_eval", split="validation", trust_remote_code=True)
    eval_dataset = eval_dataset.with_format("torch", device=device)

    # Initialize the species labels for unique species classes
    global species_dict
    uniq_species = set(train_dataset["species"])
    n_classes = len(uniq_species)
    species_dict = {entry: i for i, entry in enumerate(uniq_species)}

    # Initialize the Visual Encoder model
    model = DNAEncoder(model_name=model_handle, num_classes=n_classes)
    model.to(device)

    # Create the dataloader with the custom collate function
    collate_fn = partial(collate, tokenizer=tokenizer, species_dict=species_dict)
    train_dataloader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=64, collate_fn=collate_fn)

    # Train the model
    metrics = model.fit(train_dataloader, eval_dataloader, epochs=3, device=device)
    print(metrics)

if __name__ == "__main__":
    training_loop()

