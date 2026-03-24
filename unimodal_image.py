import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModel
import numpy as np

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VisualEncoder(nn.Module):
    def __init__(self, visual_encoder, n_classes, lr):
        super(VisualEncoder, self).__init__()
        self.visual_encoder = visual_encoder
        visual_encoder_size = 384

        self.n_classes = n_classes # Number of classes
        self.i = np.eye(n_classes) # Matrix for one-hot encoding

        # Fully connected classification head
        self.class_head = nn.Linear(visual_encoder_size, n_classes)
        self.dropout = nn.Dropout(0.1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, images):
        # CLS for embedding
        embedding = self.visual_encoder(images).last_hidden_state[:,0]

        # Pass the embedding through the model
        logits = self.class_head(embedding)

        return logits

    def train(self, dataloader, epochs):
        self.optimizer.zero_grad()

        # Loop through the epochs to train the model
        for i in range(epochs):
            self.visual_encoder.train()
            # Iterate over the batches
            for batch in dataloader:
                images = batch["images"].to(device)
                labels = batch["labels"]
                labels_onehot = torch.tensor(self.i[labels]).to(device)
                labels = labels.to(device)

                logits = self.forward(images=images)
                loss = self.criterion(logits, labels_onehot)

                loss.backward()

                self.optimizer.step()

def collate(batch):
    """Custom collation function for dataloader, extract images from the batch and pad them with the largest width from the widest image.

    Args:
        batch(dict): The batch containing the data subset

    Returns:
        torch.tensor: The stacked tensor of the batch of images.
    """
    labels = [species_dict[i["species"]] for i in batch]
    images = [i["image"] for i in batch]
    max_width = max(img.shape[-1] for img in images)
    padded = [F.pad(img, (0, max_width-img.shape[-1], 0, 0)) for img in images]
    return {"images": torch.stack(padded), "labels": torch.Tensor(labels).to(torch.uint16)}

if __name__ == "__main__":
    # Load the BIOSCAN5M dataset
    dataset = load_dataset("dataset.py", name="cropped_256_train", split="train", trust_remote_code=True)
    dataset = dataset.with_format("torch", device=device)

    # Initialize the species labels
    global species_dict
    uniq_species = set(dataset["species"])
    n_classes = len(uniq_species)
    species_dict = {entry: i for i, entry in enumerate(uniq_species)}

    visual_encoder = AutoModel.from_pretrained('facebook/dinov2-small')
    
    model = VisualEncoder(visual_encoder, n_classes, 0.5)
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=64, collate_fn=collate)

    model.train(dataloader, epochs=3)