import torch
# print("CUDA available:", torch.cuda.is_available())
# print("CUDA version:", torch.version.cuda)
# print("Device count:", torch.cuda.device_count())

import os
import pandas as pd
import numpy as np
import time
import json

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import AutoModel
from transformers import DataCollatorForLanguageModeling
from transformers import get_linear_schedule_with_warmup

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from datasets import load_dataset
from functools import partial
from dotenv import load_dotenv

load_dotenv()
apitoken = os.getenv("API_KEY")

global device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DNAEncoder(nn.Module):
    def __init__(self, model_name, num_classes, training_mode=True, lr=2e-5):
        super().__init__()

        self.training_mode = training_mode
        # Load pretrained DNA-BERT Model
        self.bert = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        # Freeze Parameters if training mode is off
        if not self.training_mode:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Create projection for to make BERT features better suited for classification
        self.projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.class_head = nn.Linear(512, num_classes) # Maps embedding to species classes
        self.criterion = nn.CrossEntropyLoss() # Indicates how bad the model is

    def encode(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Mean Pooling
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()

        embedding = (hidden * mask).sum(dim=1)
        embedding = embedding / mask.sum(dim=1).clamp(min=1e-6)

        embedding = self.projection(embedding)

        return embedding

    def forward(self, input_ids, attention_mask):
        embedding = self.encode(input_ids, attention_mask)
        logits = self.class_head(embedding)

        return logits

    def evaluate(self, eval_dataloader):
        # Evaluation
        self.eval() # Set model to evaluation mode
        eval_loss = []
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad(): # No gradient calculation (faster and less memory)
            for batch in eval_dataloader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, dtype=torch.long, non_blocking=True)

                logits = self.forward(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                eval_loss.append(loss.item())

                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        # Calculate and append Metrics
        avg_eval_loss = sum(eval_loss) / len(eval_loss)
        accuracy = correct_predictions / total_predictions

        return accuracy, avg_eval_loss
    
    def fit(self, dataloader, eval_dataloader, epochs, optimizer):
        """Custom training loop for fine-tuning"""
        metrics = {"train_loss": [], "eval_loss": [], "accuracy": []}
        best_acc = 0

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=len(dataloader) * epochs
        )
        scaler = torch.amp.GradScaler()

        for epoch in range(epochs):
            self.train() # Enable dropout
            train_loss = []

            for batch_idx, batch in enumerate(dataloader):
                optimizer.zero_grad(set_to_none=True)

                # Get input_ids, attention_mask and labels from batch
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                # Create logits and calculate loss value
                with autocast():
                    logits = self.forward(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)

                # Sanity checks every 50 training steps
                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss {loss.item():.4f}") 
                

                # Backpropagation
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # Prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                # Update weights
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # Track loss
                train_loss.append(loss.item())

            avg_train_loss = sum(train_loss) / len(train_loss)
            metrics["train_loss"].append(avg_train_loss)

            accuracy, avg_eval_loss = self.evaluate(eval_dataloader)
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(self.state_dict(), "best_model.pt")
            metrics["eval_loss"].append(avg_eval_loss)
            metrics["accuracy"].append(accuracy)
            print(f"Concluded epoch: {epoch}")
            
        return metrics
    
    def save(self, metrics, path):
        with open("train_log.json", "w") as f:
            json.dump(metrics, f)

        torch.save(self.state_dict(), path)
    

    # def load(self, path):
    #     self.load_state_dict(torch.load(path, map_location=device))
        
def leakage_check(train_dataset, eval_dataset):
    train_set = set(train_dataset["dna_barcode"])
    eval_set = set(eval_dataset["dna_barcode"])

    overlap = len(train_set.intersection(eval_set))
    return overlap

def random_kmer_crop(sequence, k=3, max_length=512):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

    if len(kmers) <= max_length:
        return " ".join(kmers)

    start = np.random.randint(0, len(kmers) - max_length)
    chunk = kmers[start:start + max_length]
    return " ".join(chunk)

def center_kmer_crop(sequence, k=3, max_length=512):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

    if len(kmers) <= max_length:
        return " ".join(kmers)

    start = (len(kmers) - max_length) // 2
    return " ".join(kmers[start:start + max_length])

def preprocess_dataset(dataset, tokenizer, species_dict, k=3, max_length=512, train=True):

    def preprocess(batch):
        sequences = []

        for seq in batch["dna_barcode"]:

            kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]

            if len(kmers) > max_length:
                start = np.random.randint(0, len(kmers) - max_length)
                kmers = kmers[start:start+max_length]

            sequences.append(" ".join(kmers))

        tokens = tokenizer(
            sequences,
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

        labels = [species_dict[s] for s in batch["species"]]
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": labels
        }

    dataset = dataset.map(
        preprocess,
        batched=True,
        batch_size=1000,
        num_proc=min(4, os.cpu_count())
    )

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    return dataset


def training_loop(lr=1e-4, batch_size=16, num_workers=4, epochs=3):
    model_handle = 'zhihan1996/DNA_bert_3'

    # Set up Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_handle, trust_remote_code=True)
    print("Tokenizer loaded")

    # Load the BIOSCAN5M train dataset
    train_dataset = load_dataset(
        "dataset.py",
        name="cropped_256_train",
        split="train",
        trust_remote_code=True,
        token=apitoken,
        cache_dir="/data/s4501888/hf/datasets"
    )
    train_dataset = train_dataset.with_format("torch")
    print("Train dataset loaded")

    # Test with random labels
    # species = train_dataset["species"]
    # np.random.shuffle(species)
    # train_dataset = train_dataset.remove_columns("species")
    # train_dataset = train_dataset.add_column("species", species)

    # Load the BIOSCAN5M validation dataset
    eval_dataset = load_dataset(
        "dataset.py",
        name="cropped_256_eval",
        split="validation",
        trust_remote_code=True,
        token=apitoken,
        cache_dir="/data/s4501888/hf/datasets"
    )
    eval_dataset = eval_dataset.with_format("torch")
    print("Evaluation dataset loaded")

    # Leakage Check
    print("Exact overlap:", leakage_check(train_dataset, eval_dataset))
    

    # Initialize the species labels for unique species classes
    uniq_species_train = set(train_dataset["species"])
    uniq_species_eval = set(eval_dataset["species"])
    uniq_species_total = uniq_species_train.union(uniq_species_eval)
    
    n_classes = len(uniq_species_total)
    uniq_species_total = sorted(list(uniq_species_total))
    species_dict = {str(entry): i for i, entry in enumerate(uniq_species_total)}
    print("Species dictionary created.")

    # Number of classes check
    print("Number of classes:", n_classes)

    # Initialize the DNA Encoder model
    model = DNAEncoder(model_name=model_handle, num_classes=n_classes)
    # model = DNAEncoder(model_name=model_handle, num_classes=n_classes, training_mode=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    print("Model initialized")
    
    # Preprocess and tokenize dataset:
    train_dataset = preprocess_dataset(
        train_dataset,
        tokenizer,
        species_dict,
        k=3,
        max_length=512
    )
    eval_dataset = preprocess_dataset(
        eval_dataset,
        tokenizer,
        species_dict,
        k=3,
        max_length=512
    )
    print("Dataset pre-processed")

    # Create the dataloader with the custom collate function
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    print("Datasets collated")

    # Train the model
    print("Training model...")
    metrics = model.fit(train_dataloader, eval_dataloader, epochs=epochs, optimizer=optimizer)
    print("Model Trained:")
    
    print(metrics)

    # Save model weights for later
    filepath = f"model_weights{time.strftime('%Y%m%d')}.weights"
    model.save(metrics, filepath)
    print(f"Model weights saved to {filepath}")

if __name__ == "__main__":
    training_loop()
    