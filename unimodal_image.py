import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModel, set_seed, AutoImageProcessor
from dotenv import load_dotenv
import os
import numpy as np
import random
import matplotlib.pyplot as plt

load_dotenv()
apitoken =os.getenv("API_KEY")

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the Encoder image processor
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small',
                                               cache_dir="/data/s4514998/hf/datasets")

class VisualEncoder(nn.Module):
    def __init__(self, visual_encoder, params):
        super(VisualEncoder, self).__init__()
        self.visual_encoder = visual_encoder
        visual_encoder_size = 384 # Size visual encoder

        # Set the Model parameters
        self.lr = params["lr"]
        self.n_classes = params["n_classes"]
        self.epochs = params["epochs"]
        self.steps_per_epoch = params["steps_per_epoch"]
        self.batch_size = params["batch_size"]

        # Set the Fully connected classification head
        self.class_head = nn.Sequential(
            nn.Linear(visual_encoder_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128), 
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, self.n_classes)
            )
        self.dropout = nn.Dropout(0.1) # Dropout layer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr) # Adam optimizer
        self.criterion = nn.CrossEntropyLoss() # Loss function for classification

        # Save the model metrics
        self.train_loss = np.zeros(shape=[self.epochs + 1, self.steps_per_epoch])
        self.eval_loss = np.zeros(shape=[self.epochs + 1, self.steps_per_epoch])
        self.train_acc = np.zeros(self.epochs)
        self.eval_acc = np.zeros(self.epochs)

    def forward(self, images):
        # Obtain the CLS for embedding
        outputs = self.visual_encoder(pixel_values=images)
        embedding = outputs.last_hidden_state[:, 0]
        # Mean pooling
        #embedding = outputs.last_hidden_state.mean(dim=1)

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

            # Initialize prev and the train loss, and the training metrics
            prev = 0
            train_loss = []
            train_correct = 0
            train_total = 0

            for timestep, idx in enumerate(range(self.batch_size, len(train_dataset), self.batch_size)):
                self.optimizer.zero_grad() # Reset the gradients

                # Collate the dataset to get the batches needed to train the model
                batch = collate(train_dataset[prev:idx])
                    
                # Get the images and labels from the current batch
                images = batch["images"]
                labels = batch["labels"]

                logits = self.forward(images=images)
                loss = self.criterion(logits, labels)

                # Compute the training metrics
                prediction = torch.argmax(logits, dim=1)
                train_correct += (prediction == labels).sum().item()
                train_total += labels.size(0)

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

            # Calculate accuracy for the training dataset
            train_accuracy = train_correct / train_total

            # Set the model to evaluation mode
            self.eval()

            # Initialize prev and the train loss, and the training metrics
            prev = 0
            eval_loss = []
            eval_prediction = 0
            eval_total = 0

            with torch.no_grad():  # Disable gradient computation for evaluation
                for timestep, idx in enumerate(range(self.batch_size, len(eval_dataset), self.batch_size)):
                    # Collate the dataset to get the batches needed to evaluate the model
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
                    eval_prediction += (prediction == labels).sum().item()
                    eval_total += labels.size(0)

                    # Break the loop when the set number of training steps are reached
                    if timestep == self.steps_per_epoch - 1:
                        break
                    
                    # Reset prev
                    prev = idx

            # Calculate accuracy for the validation dataset
            eval_accuracy = eval_prediction / eval_total

            # Save and print the metrics for the current epoch
            self.train_loss[epoch, :] = np.array(train_loss)
            self.train_acc[epoch] = np.array(train_accuracy)
            self.eval_loss[epoch, :] = np.array(eval_loss)
            self.eval_acc[epoch] = np.array(eval_accuracy)
            print("Training loss: ", self.train_loss.mean(axis=1)[epoch])
            print("Training accuracy:", train_accuracy)
            print("Validation loss: ", self.eval_loss.mean(axis=1)[epoch])
            print("Validation accuracy: ", eval_accuracy)
        
        # Print the training and evaluation loss log
        print("Training loss log: ", self.train_loss.mean(axis=1)[:-1])
        print("Validation loss log: ", self.eval_loss.mean(axis=1)[:-1])

    def plot_metrics(self, save_path):
        # Set the number of epochs used for the plots
        epochs = range(1, self.epochs + 1)

        # Plot the loss metrics for the model training and validation and save the figure
        fig, ax = plt.subplots()
        ax.plot(epochs, self.train_loss.mean(axis=1)[:-1], label="Train Loss")
        ax.plot(epochs, self.eval_loss.mean(axis=1)[:-1], label="Validation Loss")
        ax.set(xlabel="Epochs", ylabel="Loss", title="Training vs Validation Loss")
        ax.legend()
        fig.savefig(f"{save_path}_loss.png")

        # Plot the accuracy metrics for the model training and validation and save the figure
        fig, ax = plt.subplots()
        ax.plot(epochs, self.train_acc, label="Train Accuracy")
        ax.plot(epochs, self.eval_acc, label="Validation Accuracy")
        ax.set(xlabel="Epochs", ylabel="Accuracy", title="Training vs Validation Accuracy")
        ax.legend()
        fig.savefig(f"{save_path}_accuracy.png")
  
    def save(self, path):
        # Save the weights
        torch.save(self.state_dict(), path)

        # Save the metrics in one file
        metrics = {"train_loss": self.train_loss,
                   "train_acc": self.train_acc,
                   "eval_loss" : self.eval_loss,
                   "eval_acc": self.eval_acc,
                   "epochs": self.epochs
        }
        np.save(path + "_cls_species.npy", metrics)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))


def collate(batch):
    """Custom collation function for the dataset, extract images from the batch and process them with the AutoImageProcessor.

    Args:
        batch(dict): The batch containing the data subset

    Returns:
        torch.tensor: The stacked tensor of the batch of images.
        labels: The stacked tensor of the corresponding labels of the batch of images.
    """
    images = batch["image"]
    labels = [species_dict[i] for i in batch["species"]]

    # Use the model AutoImageProcessor to process the images for a better representation
    inputs = processor(images=images, return_tensors="pt")

    return {"images": inputs["pixel_values"].to(device), 
            "labels": torch.Tensor(labels).long().to(device)}

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
        batch_size=32,
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

    # Plot the metrics of the model
    model.plot_metrics(save_path="/local/mmeb_s4514998/metrics")