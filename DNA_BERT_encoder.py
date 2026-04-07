import torch
import os
import pandas as pd
import numpy as np

from time import time
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import AutoModel
from transformers import DataCollatorForLanguageModeling

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets import load_dataset
from functools import partial
from dotenv import load_dotenv

load_dotenv()
apitoken = os.getenv("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")

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
            nn.ReLU()
        )

        self.class_head = nn.Linear(512, num_classes) # Maps embedding to species classes
        self.criterion = nn.CrossEntropyLoss() # Indicates how bad the model is

    def encode(self, input_ids, attention_mask):
        """Use DNA-BERT to encode barcodes"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:, 0, :]
        embedding = self.projection(cls)
        embedding = nn.functional.normalize(embedding, dim=-1)

        return embedding

    def forward(self, input_ids, attention_mask):
        """Create Logits for classification"""
        
        # compute embedding
        embedding = self.encode(input_ids, attention_mask)
        logits = self.class_head(embedding)

        return logits

    
    def fit(self, dataloader, eval_dataloader, epochs, device, optimizer):
        """Custom training loop"""
        metrics = {"train_loss": [], "eval_loss": [], "accuracy": []}
        for epoch in range(epochs):
            self.train() # Enable dropout
            train_loss = 0

            for batch in dataloader: 

                # Get input_ids, attention_mask and labels from batch
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Reset gradients 
                optimizer.zero_grad() 

                # Create logits and calculate loss value
                logits = self.forward(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                # Backpropagation
                loss.backward()

                # Prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                # Update weights
                optimizer.step()

                # Track loss
                train_loss += loss.item()

            avg_train_loss = train_loss / len(dataloader)
            metrics["train_loss"].append(avg_train_loss)

            # Evaluation
            self.eval()
            eval_loss = 0
            correct = 0
            total = 0
            with torch.no_grad(): # No gradient calculation (faster and less memory)
                for batch in eval_dataloader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    logits = self.forward(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)

                    eval_loss += loss.item()

                    predictions = torch.argmax(logits, dim=1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

            # Calculate and append Metrics
            avg_eval_loss = eval_loss / len(eval_dataloader)
            accuracy = correct / total
            metrics["eval_loss"].append(avg_eval_loss)
            metrics["accuracy"].append(accuracy)
            print(f"Concluded epoch: {epoch}")
        return metrics
    
    def save(self, path):
        np.save("train_log.npy", self.train_loss)
        torch.save(self.state_dict(), path)
    

    def load(self, path):
        self.load_state_dict(torch.load(path))
        
    
def collate(batch, tokenizer, species_dict, k=3, max_length=512):
    """Converts raw dataset in model input"""
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

def save_checkpoint(model, optimizer, step):
    os.makedirs("checkpoints", exist_ok=True)

    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, f"checkpoints/dna_bert_step_{step}.pt")


def training_loop():
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


    # Initialize the species labels for unique species classes
    global species_dict
    uniq_species = set(train_dataset["species"])
    n_classes = len(uniq_species)
    species_dict = {entry: i for i, entry in enumerate(uniq_species)}
    print("Get unique species")

    # Initialize the DNA Encoder model
    model = DNAEncoder(model_name=model_handle, num_classes=n_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.to(device)
    print("Model initialized")

    # Create the dataloader with the custom collate function
    collate_fn = partial(collate, tokenizer=tokenizer, species_dict=species_dict, k=3)
    train_dataloader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=64, collate_fn=collate_fn)
    print("Datasets collated")

    # Train the model
    print("Evaluating model...")
    metrics = model.fit(train_dataloader, eval_dataloader, epochs=3, device=device, optimizer=optimizer)
    print("Model evaluated:")
    print(metrics)

    # Save model weights for later
    filepath = f"model_weights{time.strftime("%Y%m%d")}.weights"
    model.save(filepath)
    print(f"Model weights saved to {filepath}")

if __name__ == "__main__":
    training_loop()

