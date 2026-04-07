import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModel, set_seed
from dotenv import load_dotenv
import os
import numpy as np
import random

load_dotenv()
apitoken =os.getenv("API_KEY")

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VisualEncoder(nn.Module):
    def __init__(self, visual_encoder, params):
        super(VisualEncoder, self).__init__()
        self.visual_encoder = visual_encoder
        visual_encoder_size = 384 # Size visual encoder

        # Model parameters
        self.lr = params["lr"]
        self.n_classes = params["n_classes"]
        self.epochs = params["epochs"]
        self.steps_per_epoch = params["steps_per_epoch"]
        self.batch_size = params["batch_size"]

        # Fully connected classification head
        self.class_head = nn.Linear(visual_encoder_size, self.n_classes)
        self.dropout = nn.Dropout(0.1) # Dropout layer

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr) # Adam optimizer
        self.criterion = nn.CrossEntropyLoss() # Loss function for classification

        # Saving the model metrics
        self.train_loss = np.zeros(shape=[self.epochs + 1, self.steps_per_epoch])
        self.eval_loss = np.zeros(shape=[self.epochs + 1, self.steps_per_epoch])

    def forward(self, images):
        # CLS for embedding
        embedding = self.visual_encoder(images).last_hidden_state[:,0]

        # Pass the embedding through the model
        embedding = self.dropout(embedding)
        logits = self.class_head(embedding)

        return logits

    def fit(self, train_dataset, eval_dataset):
        # Loop through the epochs to train the model
        for epoch in range(self.epochs):
            # Print the current epoch
            print(f"Epoch {epoch+1}/{self.epochs}")

            # Set the model to train mode
            self.train()

            # Shuffle the dataset at the start for more variable data training
            train_dataset = train_dataset.shuffle()

            # Initialize prev and the train loss
            prev = 0
            train_loss = []

            for timestep, idx in enumerate(range(self.batch_size, len(train_dataset), self.batch_size)):
                self.optimizer.zero_grad() # Reset the gradients

                # Collate the dataset to get the batches needed to train the model
                batch = collate(train_dataset[prev:idx])
                    
                # Get the images and labels from the current batch
                images = batch["images"]
                labels = batch["labels"]

                logits = self.forward(images=images)
                loss = self.criterion(logits, labels)

                # Backward pass
                loss.backward() 
                self.optimizer.step()

                # Save the loss
                train_loss.append(loss.item())

                # Break the training loop when the set number of training steps are reached
                if timestep == self.steps_per_epoch - 1:
                    break
                
                # Reset prev
                prev = idx

            # Validation step metrics
            self.eval()  # Set the model to evaluation mode
            correct_predictions = 0
            total_predictions = 0
            prev = 0
            eval_loss = []

            with torch.no_grad():  # Disable gradient computation for evaluation
                for timestep, idx in enumerate(range(self.batch_size, len(eval_dataset), self.batch_size)):
                    # Collate the dataset to get the batches needed to train the model
                    batch = collate(eval_dataset[prev:idx])
                        
                    # Get the images and labels from the current batch
                    images = batch["images"]
                    labels = batch["labels"]

                    logits = self.forward(images=images)
                    loss = self.criterion(logits, labels)

                    # Save the loss
                    eval_loss.append(loss.item())

                    # Compute metrics
                    prediction = torch.argmax(logits, dim=1)
                    correct_predictions += (prediction == labels).sum().item()
                    total_predictions += labels.size(0)

                    # Break the training loop when the set number of training steps are reached
                    if timestep == self.steps_per_epoch - 1:
                        break
                    
                    # Reset prev
                    prev = idx

            # Calculate accuracy for the validation dataset
            accuracy = correct_predictions / total_predictions

            # Save and print the metrics for the current epoch
            self.train_loss[epoch, :] = np.array(train_loss)
            self.eval_loss[epoch, :] = np.array(eval_loss)
            print("Training loss: ", self.train_loss.mean(axis=1)[epoch])
            print("Validation loss: ", self.eval_loss.mean(axis=1)[epoch])
            print("Validation accuracy: ", accuracy)
        
        # Print the training and evaluation log
        print("Train log: ", self.train_loss.mean(axis=1))
        print("Validation log: ", self.eval_loss.mean(axis=1))

    def save(self, path):
        np.save("train_log.npy", self.train_loss)
        torch.save(self.state_dict(), path)
    

    def load(self, path):
        self.load_state_dict(torch.load(path))

def collate(batch):
    """Custom collation function for the dataset, extract images from the batch and pad them with the largest width from the widest image.

    Args:
        batch(dict): The batch containing the data subset

    Returns:
        torch.tensor: The stacked tensor of the batch of images.
    """
    images = batch["image"]
    labels = [species_dict[i] for i in batch["species"]]
    max_width = max(img.shape[-1] for img in images)
    padded = [F.pad(img, (0, max_width-img.shape[-1], 0, 0)) for img in images]
    return {"images": torch.stack(padded).to(device), "labels": torch.Tensor(labels).long().to(device)}

if __name__ == "__main__":
    # Set the seed for everything to be deterministic
    seed = 202667
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the BIOSCAN5M train dataset
    train_dataset = load_dataset("dataset.py", 
                                 name="cropped_256_train", 
                                 split="train", 
                                 trust_remote_code=True, 
                                 cache_dir="/data/s4514998/hf/datasets")
    train_dataset = train_dataset.with_format("torch", device=device)

    # Load the BIOSCAN5M validation dataset
    eval_dataset = load_dataset("dataset.py", 
                                name="cropped_256_eval", 
                                split="validation", 
                                trust_remote_code=True, 
                                cache_dir="/data/s4514998/hf/datasets")
    eval_dataset = eval_dataset.with_format("torch", device=device)

    # Initialize the species labels for the unique species classes
    uniq_species = set(train_dataset["species"])
    n_classes = len(uniq_species)
    species_dict = {entry: i for i, entry in enumerate(uniq_species)}

    # Initialize the parameters used for the model
    parameters = dict(
        lr = 5e-5,
        epochs=200,
        steps_per_epoch=200,
        batch_size=16,
        n_classes=n_classes
    )

    # Initialize the Visual Encoder model
    visual_encoder = AutoModel.from_pretrained('facebook/dinov2-small', 
                                               cache_dir="/data/s4514998/hf/datasets")
    model = VisualEncoder(visual_encoder, params=parameters)
    model.to(device)

    # Train the model
    model.fit(train_dataset=train_dataset, eval_dataset=eval_dataset)

    # Save the model
    model.save("/local/mmeb_s4514998/model.weights")