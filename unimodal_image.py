import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModel

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VisualEncoder(nn.Module):
    def __init__(self, visual_encoder, n_classes, lr):
        super(VisualEncoder, self).__init__()
        self.visual_encoder = visual_encoder
        self.n_classes = n_classes # Number of classes
        visual_encoder_size = 384 # Size visual encoder

        # Fully connected classification head
        self.class_head = nn.Linear(visual_encoder_size, n_classes)
        self.dropout = nn.Dropout(0.1) # Dropout layer

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr) # Adam optimizer
        self.criterion = nn.CrossEntropyLoss() # Loss function for classification

    def forward(self, images):
        # CLS for embedding
        embedding = self.visual_encoder(images).last_hidden_state[:,0]

        # Pass the embedding through the model
        embedding = self.dropout(embedding)
        logits = self.class_head(embedding)

        return logits

    def fit(self, dataloader, eval_dataloader, epochs):
        # Initialize the model metrics dict
        metrics = {
            "train_loss": [],
            "eval_loss": [],
            "accuracy": []
        }

        # Loop through the epochs to train the model
        for epoch in range(epochs):
            # Set the model to train mode
            self.train()

            # Initialize total loss and predictions
            train_loss = 0

            # Iterate over the batches
            for batch in dataloader:
                self.optimizer.zero_grad() # Reset the gradients
                
                # Get the images and labels from the current batch
                images = batch["images"]
                labels = batch["labels"]

                logits = self.forward(images=images)
                loss = self.criterion(logits, labels)

                # Backward pass
                loss.backward() 
                self.optimizer.step()

                # Track the loss
                train_loss += loss.item()

            # Calculate the average training loss
            avg_train_loss = train_loss / len(dataloader)

            # Save the metrics
            metrics["train_loss"].append(avg_train_loss)

            # Validation step metrics
            self.eval()  # Set the model to evaluation mode
            eval_loss = 0.0
            correct = 0
            total_predictions = 0

            with torch.no_grad():  # Disable gradient computation for evaluation
                for batch in eval_dataloader:

                    # Get the images and labels from the current batch
                    images = batch["images"]
                    labels = batch["labels"]

                    logits = self.forward(images=images)
                    loss = self.criterion(logits, labels)

                    # Tracking the loss
                    eval_loss += loss.item()

                    # Compute metrics
                    predict = torch.argmax(logits, dim=1)
                    correct += (predict == labels).sum().item()
                    total_predictions += labels.size(0)

            # Calculate average loss and accuracy
            avg_eval_loss = eval_loss / len(eval_dataloader)
            accuracy = correct / total_predictions

            # Save the metrics
            metrics["eval_loss"].append(avg_eval_loss)
            metrics["accuracy"].append(accuracy)

            # Print the metrics for the current epoch
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.4f}")

        return metrics

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
    return {"images": torch.stack(padded).to(device), "labels": torch.Tensor(labels).long().to(device)}

if __name__ == "__main__":
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
    visual_encoder = AutoModel.from_pretrained('facebook/dinov2-small')
    model = VisualEncoder(visual_encoder, n_classes, 1e-4)
    model.to(device)

    # Create the dataloader with the custom collate function
    train_dataloader = DataLoader(train_dataset, batch_size=64, collate_fn=collate)
    eval_dataloader = DataLoader(eval_dataset, batch_size=64, collate_fn=collate)

    # Train the model
    metrics = model.fit(train_dataloader, eval_dataloader, epochs=3)
    print(metrics)