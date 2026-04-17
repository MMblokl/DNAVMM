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
from sklearn.metrics import f1_score

load_dotenv()
apitoken =os.getenv("API_KEY")

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VisualEncoder(nn.Module):
    def __init__(self, 
                 params: dict,
                v_enc: str = "facebook/dinov2-small",
                i_processor: str = "facebook/dinov2-small",
                cache_dir: str | bool = False
                ):
        super(VisualEncoder, self).__init__()
        # Determine wheter to use the pre-defined cache directory for the parameters and data
        if cache_dir:
            self.visual_encoder = AutoModel.from_pretrained(
                v_enc,
                token=apitoken,
                cache_dir=cache_dir
            )
            self.i_processor = AutoImageProcessor.from_pretrained(
                i_processor,
                token=apitoken,
                cache_dir=cache_dir
            )

        else:
            self.visual_encoder = AutoModel.from_pretrained(
                v_enc,
                token=apitoken
            )
            self.i_processor = AutoImageProcessor.from_pretrained(
                i_processor,
                token=apitoken
            )

        # Initialize the parameters
        self.lr = params["lr"]

        self.class_values = params["class_values"]
        self.class_mapping = params["class_mapping"]
        self.epoch_ordering = params["epoch_ordering"]
        self.layer_freezing = params["layer_freezing"]

        self.steps_per_epoch = params["steps_per_epoch"]
        self.batch_size = params["batch_size"]

        # Initialize the total epochs
        total_epochs = sum(self.epoch_ordering.values())

        # Initialize the hardcoded output size of DINOV2_small
        v_enc_size = 384

        # Set the Fully connected classification head
        self.class_head = nn.Sequential(
            nn.Linear(v_enc_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128), 
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, [i for i in self.class_values.values()][0])
            )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        # Save the model metrics
        self.train_loss = np.zeros(shape=[total_epochs + 1, self.steps_per_epoch])
        self.eval_loss = np.zeros(shape=[total_epochs + 1, self.steps_per_epoch])
        self.train_acc = 0
        self.eval_acc = 0
        self.train_f1 = 0
        self.eval_f1 = 0

    def collate_fn(self, batch):
        """Custom collation function for the dataset, extract images from the batch and process them with the AutoImageProcessor.

        Args:
            batch(dict): The batch containing the data subset

        Returns:
            torch.tensor: The stacked tensor of the batch of images.
            labels: The stacked tensor of the corresponding labels of the batch of images.
        """
        images = batch["image"]
        images = self.i_processor(images=images, return_tensors="pt").to(device)
        labels = [self.class_mapping[self.labeltype][i] for i in batch[self.labeltype]]
        
        return {"images": images,
                "labels": torch.tensor(labels).long().to(device)}
    
    def forward(self, images):
        # CLS for embedding
        embedding = self.visual_encoder(**images).last_hidden_state[:,0]

        # Mean pooling for embedding
        #embedding = self.visual_encoder(**images).last_hidden_state.mean(dim=1)

        # Pass the embedding through the model
        logits = self.class_head(embedding)

        return logits

    def fit(self, train_dataset):
        """Fit the model on a shuffled batch of the dataset"""
        self.train() # Turn on the dropouts
        prev = 0   # Previous index of the dataset
        train_loss = [] # List for the training loss
        train_correct = 0 # The number of correct predictions
        train_total = 0 # Total number of training labels
        prediction_list = [] # List for all predictions done per epoch
        labels_list = [] # List for all labels per epoch

        for timestep, idx in enumerate(range(self.batch_size, len(train_dataset), self.batch_size)):
            self.optimizer.zero_grad() # Zero out previous grad

            # Collate the dataset to get the batches needed to train the model
            batch = self.collate_fn(train_dataset[prev:idx])

            # Get the images and labels from the current batch
            images = batch["images"]
            labels = batch["labels"]

            # Calculate the loss
            logits = self.forward(images=images)
            loss = self.criterion(logits, labels)

            # Compute the training metrics
            prediction = torch.argmax(logits, dim=1)
            train_correct += (prediction == labels).sum().item()
            train_total += labels.size(0)
            prediction_list.extend(prediction.cpu().detach().numpy())
            labels_list.extend(labels.cpu().detach().numpy())

            # Backward pass
            loss.backward() 
            self.optimizer.step()

            # Save the loss
            train_loss.append(loss.item())

            # Break the training loop when the set number of training steps are reached
            if timestep == self.steps_per_epoch - 1:
                break
                
            # Reset the previous index of the dataset
            prev = idx
        
        return train_correct, train_total, train_loss, prediction_list, labels_list
    

    def evaluate(self, eval_dataset):
        """Single evaluation run on the validation dataset for each epoch."""
        self.eval() # Turn off the dropouts
        prev = 0   # Previous index of the dataset
        eval_loss = [] # List for the validation loss
        eval_correct = 0 # The number of correct predictions
        eval_total = 0 # Total number of validation labels
        prediction_list = [] # List for all predictions done per epoch
        labels_list = [] # List for all labels per epoch
  
        with torch.no_grad():  # Disable gradient computation for evaluation
            for timestep, idx in enumerate(range(self.batch_size, len(eval_dataset), self.batch_size)):

                # Collate the dataset to get the batches needed to evaluate the model
                batch = self.collate_fn(eval_dataset[prev:idx])

                # Get the images and labels from the current batch
                images = batch["images"]
                labels = batch["labels"]

                # Calculate the loss
                logits = self.forward(images=images)
                loss = self.criterion(logits, labels)

                # Compute the validation metrics
                prediction = torch.argmax(logits, dim=1)
                eval_correct += (prediction == labels).sum().item()
                eval_total += labels.size(0)
                prediction_list.extend(prediction.cpu().detach().numpy())
                labels_list.extend(labels.cpu().detach().numpy())

                # Save the loss
                eval_loss.append(loss.item())

                # Break the validation loop when the set number of training steps are reached
                if timestep == self.steps_per_epoch - 1:
                    break
                    
                # Reset the previous index of the dataset
                prev = idx

        return eval_correct, eval_total, eval_loss, prediction_list, labels_list

    def freeze_until(self, model, until):
        """Freezes the weights of a given model according to the given the value. If until=1, first 2 layers are frozen."""
        for i, layer in enumerate(model.encoder.layer):
            # If the index is with in the to be frozen layers
            if i <= until:
                for param in layer.parameters():
                    param.requires_grad = False

    def update_optimizer(self):
        """Updates the optimizer to only use the trainable model params"""
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.lr)

    def train_loop(self, train_dataset, eval_dataset):
        """Loop through the class and label types"""
        for labeltype in self.class_values.keys():
            self.labeltype = labeltype # Set the current label type
            epochs = self.epoch_ordering[labeltype] # Get the number of epochs for the label
            
            until = self.layer_freezing[self.labeltype] # How many layers to freeze
            if until: # If None, no freeze.
                # Freeze the bottom n layers
                self.freeze_until(self.visual_encoder, until)

                # Set the optimizer to only optimise on non-frozen weights
                self.update_optimizer()

            # Replace the final class head layer with a new output layer that matches the number of classes
            self.class_head[8] = nn.Linear(128, self.class_values[self.labeltype]).to(device)

            # Epoch loop
            for epoch in range(epochs):
                # Shuffle the training dataset
                train_dataset = train_dataset.shuffle()
                print(f"Epoch {epoch+1}/{epochs} for {labeltype}.")

                # Train on the dataset for an entire epoch
                train_correct, train_total, train_loss, train_preds, train_labels = self.fit(train_dataset=train_dataset)

                # Test the validation dataset for an entire epoch
                eval_correct, eval_total, eval_loss, eval_preds, eval_labels = self.evaluate(eval_dataset=eval_dataset)

                # Save and print the training metrics for the current epoch
                self.train_loss[epoch, :] = np.array(train_loss)
                print("Training loss: ", self.train_loss.mean(axis=1)[epoch])

                # Save and print the validation metrics for the current epoch
                self.eval_loss[epoch, :] = np.array(eval_loss)
                print("Validation loss: ", self.eval_loss.mean(axis=1)[epoch])
            
        # Compute and save the accuracy after training
        self.train_acc = train_correct / train_total
        self.eval_acc = eval_correct / eval_total

        # Compute and save the F1 score after training, using macro to deal with multiclass classification
        self.train_f1 = f1_score(train_labels, train_preds, average="macro")
        self.eval_f1 = f1_score(eval_labels, eval_preds, average="macro")

        print(f"Training accuracy: {self.train_acc} Training F1 score: {self.train_f1}")
        print(f"Training accuracy: {self.eval_acc} Training F1 score: {self.eval_f1}")

        # Print the final training log that shows the loss of each epoch
        print("Training loss log: ", self.train_loss.mean(axis=1)[:-1])

    def plot_metrics(self, save_path):
        # Set the number of epochs used for the plots
        epochs = range(1, self.epoch_ordering["species"]+1)
  
        # Plot the loss metrics for the model training and validation and save the figure
        fig, ax = plt.subplots()
        ax.plot(epochs, self.train_loss.mean(axis=1)[:-1][0:self.epoch_ordering["species"]], label="Train Loss")
        ax.plot(epochs, self.eval_loss.mean(axis=1)[:-1][0:self.epoch_ordering["species"]], label="Validation Loss")
        ax.set(xlabel="Epochs", ylabel="Loss", title="Training vs Validation Loss")
        ax.legend()
        fig.savefig(f"{save_path}_loss.png")

    def save(self, path):
        # Save the weights
        torch.save(self.state_dict(), path)

        # Save the metrics in one file
        metrics = {"train_loss": self.train_loss,
                   "train_acc": self.train_acc,
                   "train_f1":  self.train_f1,
                   "eval_loss" : self.eval_loss,
                   "eval_acc": self.eval_acc,
                   "eval_f1": self.eval_f1,
                   "epochs": [i for i in self.epoch_ordering.values()],
                   "frozen_until": [i for i in self.layer_freezing.values()]
        }
        np.save(path + "_cls_species.npy", metrics)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))

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

    # Set the cache directory location
    cache_dir = "/data/s4514998/hf/datasets"

    # Enable hierarchical training, False for singular species level
    hierachical = True

    # Load the BIOSCAN5M train dataset
    train_dataset = load_dataset(
        "dataset.py",
        name="cropped_256_train", 
        split="train", 
        trust_remote_code=True,
        token=apitoken,
        cache_dir="/data/s4514998/hf/datasets"
        )
    train_dataset = train_dataset.with_format("torch", device=device)

    # Load the BIOSCAN5M validation dataset
    eval_dataset = load_dataset(
        "dataset.py", 
        name="cropped_256_eval", 
        split="validation", 
        trust_remote_code=True, 
        token=apitoken,
        cache_dir="/data/s4514998/hf/datasets"
        )
    eval_dataset = eval_dataset.with_format("torch", device=device)

    # Initialize the taxonomic labels for organisms in the dataset
    uniq_classes = set.union(set(train_dataset["class"]), set(train_dataset["class"])) # Class taxonomic level
    class_dict = {entry: i for i, entry in enumerate(uniq_classes)}
    n_class = len(uniq_classes)

    uniq_orders = set.union(set(train_dataset["order"]), set(eval_dataset["order"])) # Order taxonomic level
    order_dict = {entry: i for i, entry in enumerate(uniq_orders)}
    n_orders = len(uniq_orders)

    uniq_families = set.union(set(train_dataset["family"]), set(eval_dataset["family"])) # Family taxonomic level
    family_dict = {entry: i for i, entry in enumerate(uniq_families)}
    n_family = len(uniq_families)

    uniq_genus = set.union(set(train_dataset["genus"]), set(eval_dataset["genus"])) # Genus taxonomic level
    genus_dict = {entry: i for i, entry in enumerate(uniq_genus)}
    n_genus = len(uniq_genus)

    uniq_species = set.union(set(train_dataset["species"]), set(train_dataset["species"])) # Species taxonomic level
    species_dict = {entry: i for i, entry in enumerate(uniq_species)}
    n_species = len(uniq_species)

    if hierachical:
        # Initialize parameters to perform hierarchical training
        parameters = dict(
            lr = 5e-5,
            steps_per_epoch=200,
            batch_size=16,
            class_values = {
                "class": n_class,
                "order": n_orders,
                "family":n_family,
                "genus": n_genus,
                "species": n_species,
                },
            class_mapping = {
                "class": class_dict,
                "order": order_dict,
                "family": family_dict,
                "genus": genus_dict,
                "species": species_dict,
                },
            epoch_ordering = {
                "class": 2,
                "order": 4,
                "family": 25,
                "genus": 50,
                "species": 200
                }, # Number of epochs for each step.
            layer_freezing = {
                "class": None,
                "order": 1,
                "family": 4,
                "genus": 5,
                "species": 9,
                }
        )
    
    else:
        # Initialize the parameters to perform singular species training
        parameters = dict(
            lr = 5e-5,
            steps_per_epoch=200,
            batch_size=16,
            class_values = {
                "species": n_species,
                },
            class_mapping = {
                "species": species_dict,
                },
            epoch_ordering = {
                "species": 200
                }, # Number of epochs for each step.
            layer_freezing = {
                "species": None,
                }
        )

    # Set the model parameters and train the model
    model = VisualEncoder(
        cache_dir=cache_dir,
        params=parameters,
    )
    model = model.to(device)
    model.train_loop(train_dataset, eval_dataset)

    # Save the model weights obtained
    model.save("/local/MM_4514998/model.weights")

    # Plot the metrics of the model
    model.plot_metrics(save_path="/local/MM_4514998/metrics")