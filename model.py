import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel
from datasets import load_dataset
import numpy as np


global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DNAVMM(nn.Module):
    def __init__(self, d_enc, v_enc, n_classes, lr):
        super(DNAVMM, self).__init__()
        self.visual_encoder = v_enc
        self.dna_encoder = d_enc
        
        d_enc_size = 786
        v_enc_size = 384

        # Create np to_one_hot for ease of use
        self.n_classes = n_classes
        self.i = np.eye(n_classes)


        # Fully connected classification head
        self.class_head = nn.Linear(v_enc_size, n_classes)
        self.dropout = nn.Dropout(0.1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
    
    # Untested, probably works
    def forward(self, images, dna):
        # Other options
        #emb1 = out1.pooler_output
        #emb1 = out1.last_hidden_state.mean(dim=1)
        
        # CLS for embedding.
        v_embedding = self.visual_encoder(images).last_hidden_state[:,0]
        d_embedding = self.dna_encoder(dna).last_hidden_state[:,0]

        # Combine 
        feature_vec = torch.cat([v_embedding, d_embedding], dim=-1)

        # Pass through model.
        feature_vec = self.dropout(feature_vec)
        logits = self.class_head(feature_vec)

        return logits
    
    # Might not be possible in same class, check RL implementations
    # Ideally, just doing model.train() will do it.
    def train(self, dataloader, epochs):
        for i in range(epochs):
            self.dna_encoder.train()
            self.visual_encoder.train()
            
            for batch in dataloader:
                self.optimizer.zero_grad()

                # Barcodes unfinished
                
                images = batch["images"]
                labels = batch["labels"]
                labels_onehot = torch.Tensor(self.i[labels])
                barcodes = batch["barcodes"]

                logits = self.forward(images=images, dna=barcodes) # Doesnt work yet, is legit just CLS into class_head
                loss = self.criterion(logits, labels_onehot)

                loss.backward()

                self.optimizer.step()


def collate_fn(batch):
    """Custom collation function for dataloader, extract images from the batch and pad them with the largest width from the widest image.

    Args:
        batch(dict): The batch containing the data subset

    Returns:
        torch.tensor: The stacked tensor of the batch of images.
    """
    labels = [species_dict[i["species"]] for i in batch]
    images = [i["image"] for i in batch]
    #barcodes = [torch.Tensor(i["dna_barcode"]) for i in batch]
    max_width = max(img.shape[-1] for img in images)
    padded = [F.pad(img, (0, max_width-img.shape[-1], 0, 0)) for img in images]
    return {"images": torch.stack(padded), "labels": torch.Tensor(labels).to(torch.uint16)}





if __name__ == "__main__":
    dataset = load_dataset("dataset.py", name="cropped_256_train", split="train", trust_remote_code=True, token="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")
    dataset = dataset.with_format("torch", device=device)

    # Initialize every single species as a valuen integer
    global species_dict
    uniq_species = set(dataset["species"])
    n_classes = len(uniq_species)
    species_dict = {entry: i for i, entry in enumerate(uniq_species)}

    d_enc = AutoModel.from_pretrained("zhihan1996/DNA_bert_6", token="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")
    v_enc = AutoModel.from_pretrained("facebook/dinov2-small", token="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")

    model = DNAVMM(d_enc, v_enc, n_classes, 0.5)
    model.to(device)



    dataloader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn)

    model.train(dataloader, epochs=3)

