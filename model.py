import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, set_seed
from datasets import load_dataset
import numpy as np
import random
import os
from dotenv import load_dotenv

load_dotenv()
apitoken = os.getenv("API_KEY")

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DNAVMM(nn.Module):
    def __init__(self, d_enc, v_enc, d_tokenizer, n_classes, lr):
        super(DNAVMM, self).__init__()
        self.visual_encoder = v_enc
        self.dna_encoder = d_enc
        self.dna_tokenizer = d_tokenizer
        
        self.n_classes = n_classes
        d_enc_size = 768
        v_enc_size = 384

        # Fully connected classification head
        self.class_head = nn.Linear(d_enc_size + v_enc_size, n_classes)
        self.dropout = nn.Dropout(0.1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        # Training metrics storage
        self.train_loss = {}


    # Untested, probably works
    def forward(self, images, dna):
        # Other options
        #emb1 = out1.pooler_output
        #emb1 = out1.last_hidden_state.mean(dim=1)
        # CLS for embedding.
        v_embedding = self.visual_encoder(images).last_hidden_state[:,0]
        d_embedding = self.dna_encoder(**dna).last_hidden_state[:,0]

        # Combine 
        feature_vec = torch.cat([v_embedding, d_embedding], dim=-1)

        # Pass through model.
        feature_vec = self.dropout(feature_vec)
        logits = self.class_head(feature_vec)

        return logits
    
    # Might not be possible in same class, check RL implementations
    # Ideally, just doing model.train() will do it.
    def fit(self, dataset, epochs, steps_per_epoch=100, batch_size=8):
        self.train() # Turn on dropouts
        # Loop through each epoch
        for epoch in range(epochs):
            self.train_loss[epoch] = []
            
            # Shuffle the dataset at the start for more variable data training
            dataset = dataset.shuffle()
            prev = 0
            for timestep, idx in enumerate(range(batch_size, len(dataset), batch_size)):
                # Collate data properly
                batch = collate_fn(dataset[prev:idx])

                self.optimizer.zero_grad() # Zero out previous grad

                # Single out data
                images = batch["images"]
                labels = batch["labels"]
                barcodes = batch["barcodes"]

                # Tokenize each k-mer in the barcode
                tokenized_barcodes = self.dna_tokenizer(barcodes, return_tensors = 'pt', padding=True).to(device)

                logits = self.forward(images=images, dna=tokenized_barcodes)
                loss = self.criterion(logits, labels)

                loss.backward()

                self.optimizer.step()
                
                # Save the loss to storage
                self.train_loss[epoch].append(loss.item())
                if timestep == steps_per_epoch:
                    break
                
                # reset prev
                prev = idx
            breakpoint()

    
    def save(self, path):
        torch.save(self.state_dict(), path)
    

    def load(self, path):
        self.load_state_dict(torch.load(path))


def collate_fn(batch):
    """Custom collation function for dataloader, extract images from the batch and pad them with the largest width from the widest image.

    Args:
        batch(dict): The batch containing the data subset

    Returns:
        torch.tensor: The stacked tensor of the batch of images.
    """
    images = batch["image"]
    labels = [species_dict[i] for i in batch["species"]]
    barcodes = [" ".join([seq[i:i+k] for i in range(len(seq) - k + 1)]) for seq in batch["dna_barcode"]]
    max_width = max(img.shape[-1] for img in images)
    padded = [F.pad(img, (0, max_width-img.shape[-1], 0, 0)) for img in images]
    return {"images": torch.stack(padded), "labels": torch.tensor(labels).long().to(device), "barcodes": barcodes}
    


if __name__ == "__main__":
    # Set seed for everything
    seed = 202667
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    dataset = load_dataset(
        "dataset.py",
        name="cropped_256_train",
        split="train",
        trust_remote_code=True,
        token=apitoken,
        cache_dir="/data/s4501888/hf/datasets"
    )
    dataset = dataset.with_format("torch", device=device)

    # Initialize every single species as a valuen integer
    uniq_species = set(dataset["species"])
    n_classes = len(uniq_species)
    species_dict = {entry: i for i, entry in enumerate(uniq_species)}
    k = 6

    d_enc = AutoModel.from_pretrained(
        "zhihan1996/DNA_bert_6",
        token=apitoken,
        cache_dir="/data/s4501888/hf/datasets"
    )
    v_enc = AutoModel.from_pretrained(
        "facebook/dinov2-small",
        token=apitoken,
        cache_dir="/data/s4501888/hf/datasets"
    )
    d_tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNA_bert_6",
        token=apitoken,
        cache_dir="/data/s4501888/hf/datasets"
    )
    
    # Double size of the model for 1024 input size of DNA
    d_tokenizer.model_max_length = 1024
    d_enc.config.max_positional_embeddings = 1024
    d_enc.base_model.embeddings.position_ids = torch.arange(1024).expand((1,-1))
    d_enc.base_model.embeddings.token_type_ids = torch.zeros(1024).expand((1,-1))
    orig_pos_emb = d_enc.base_model.embeddings.position_embeddings.weight
    d_enc.base_model.embeddings.position_embeddings.weight = torch.nn.Parameter(torch.cat((orig_pos_emb, orig_pos_emb)))

    model = DNAVMM(d_enc, v_enc, d_tokenizer, n_classes, 1e-4)
    model.to(device)

    model.fit(dataset, epochs=2, steps_per_epoch=20, batch_size=8)
    model.save("/local/mmeb_s4501888/model.weights")
