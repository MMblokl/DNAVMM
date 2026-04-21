import torch
import torch.nn as nn
from transformers import AutoModel, set_seed, AutoImageProcessor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelModule(nn.Module):
    def __init__(
            self,
            params: dict,
            run_name: str,
            ds_randomization: bool = False,
            augmentation: bool = False,
            hierarchical: bool = False
        ):
        super(ModelModule, self).__init__()
        
        # Initialize the parameters
        self.lr = params["lr"]

        self.class_values = params["class_values"]
        self.class_mapping = params["class_mapping"]
        self.epoch_ordering = params["epoch_ordering"]
        self.layer_freezing = params["layer_freezing"]

        self.steps_per_epoch = params["steps_per_epoch"]
        self.batch_size = params["batch_size"]

        # Initialize whether to use dataset randomization
        self.ds_randomization = ds_randomization
        # Initialize whether to use image augmentation
        self.augmentation = augmentation
        # Whether the training scheme is hierarchical for logging
        self.hierarchical = hierarchical

        # Save name of run
        self.run_name = run_name

        # Metrics storage
        self.total_epochs = sum(self.epoch_ordering.values())
        self.train_loss = np.zeros(shape=[self.total_epochs + 1, self.steps_per_epoch])
        self.eval_loss = np.zeros(shape=[self.total_epochs + 1, self.steps_per_epoch])
        self.train_acc = 0
        self.eval_acc = 0
        self.train_f1 = 0
        self.eval_f1 = 0


    def train_loop(self, train_dataset, eval_dataset):
        """Loop through the class and label types"""
        label_options = [i for i in self.class_values.keys()]
        
        # If loaded from checkpoint, this value is already initialized
        try:
            label_options = label_options[label_options.index(self.labeltype):]
            self.check_start = True
            epochs = self.epoch_ordering[self.labeltype]
            epoch_range = [i for i in range(epochs)][self.c_epoch:]
        except AttributeError:
            # Not a checkpoint
            self.check_start = False
        
        for labeltype in label_options:
            self.labeltype = labeltype # Set the current label type
            epochs = self.epoch_ordering[labeltype] # Get the number of epochs for the label
            
            until = self.layer_freezing[self.labeltype] # How many layers to freeze
            if until: # If None, no freeze.
                # Freeze the bottom n layers
                self.freezeencoders(until=until)

            # Make sure a checkpoint load doesnt destoy the old class head
            if not self.check_start:
                # Replace the final class head layer with a new output layer that matches the number of classes
                self.class_head[8] = nn.Linear(128, self.class_values[self.labeltype]).to(device)

                # Create epoch range for new epoch loop
                epoch_range = [i for i in range(epochs)]

                # Set the optimizer to only optimise trainable weights with requires_grad=True
                self.update_optimizer()

            # Epoch loop
            for epoch in epoch_range:
                self.c_epoch = epoch + 1
                print(f"Epoch {epoch+1}/{epochs} for {labeltype}.")

                # Check whether to perform training dataset randomization with .shuffle()
                if self.ds_randomization:   
                    # Shuffle the training dataset
                    train_dataset = train_dataset.shuffle()
                

                # Check for the final epoch when calling self.fit and self.evaluate
                # Final epoch
                if self.c_epoch == epochs:
                    # Train on the dataset for an entire epoch
                    train_correct, train_total, train_loss, train_preds, train_labels = self.fit(train_dataset=train_dataset, final=True)

                    # Test the validation dataset for an entire epoch
                    eval_correct, eval_total, eval_loss, eval_preds, eval_labels = self.evaluate(eval_dataset=eval_dataset, final=True)

                # Every epoch
                else:
                    train_loss = self.fit(train_dataset=train_dataset, final=False)
                    eval_loss = self.evaluate(eval_dataset=eval_dataset, final=False)

                # Save and print the training metrics for the current epoch
                self.train_loss[epoch, :] = np.array(train_loss)
                print("Training loss: ", self.train_loss.mean(axis=1)[epoch])
                # Save and print the validation metrics for the current epoch
                self.eval_loss[epoch, :] = np.array(eval_loss)
                print("Validation loss: ", self.eval_loss.mean(axis=1)[epoch])

                if self.c_epoch % 5 == 0:
                    self.save(f"./{self.run_name}/")
            
            # After at least 1 epoch this needs a reset to make sure the class head is replaced
            self.check_start = False
            
        # Compute and save the accuracy after training
        self.train_acc = train_correct / train_total
        self.eval_acc = eval_correct / eval_total

        # Compute and save the F1 score after training, using macro to deal with multiclass classification
        self.train_f1 = f1_score(train_labels, train_preds, average="macro")
        self.eval_f1 = f1_score(eval_labels, eval_preds, average="macro")

        print(f"Training accuracy: {self.train_acc} Training F1 score: {self.train_f1}")
        print(f"Validation accuracy: {self.eval_acc} Validation F1 score: {self.eval_f1}")

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
        torch.save({
            "model": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, f"{path}/latest.pt")

        # Save the metrics in one file
        metrics = {"current_label": self.labeltype,
                   "current_epoch": self.c_epoch,
                   "train_loss": self.train_loss,
                   "train_acc": self.train_acc,
                   "train_f1": self.train_f1,
                   "eval_loss" : self.eval_loss,
                   "eval_acc": self.eval_acc,
                   "eval_f1": self.eval_f1,
                   "epochs": self.epoch_ordering,
                   "frozen_until": self.layer_freezing,
                   "hierarchical": self.hierarchical,
                   "dataset_randomization": self.ds_randomization,
                   "augmentation": self.augmentation,
        }
        np.save(f"{path}/model_metrics.npy", metrics)


    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        # Some of the values arent kept op proper device, this is the best way to fix
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    
    def update_optimizer(self):
        """Updates the optimizer to only use the trainable model params"""
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.lr)
    
    
    def freeze_until(self, model, until):
        """Freezes the weights of a given model according to the given the value. If until=1, first 2 layers are frozen."""
        for i, layer in enumerate(model.encoder.layer):
            # If the index is with in the to be frozen layers
            if i <= until:
                for param in layer.parameters():
                    param.requires_grad = False

    def start_from_checkpoint(self, path):
        # Load saved metrics
        metrics = np.load(f"{path}/model_metrics.npy", allow_pickle=True).item()

        self.train_loss = metrics["train_loss"]
        self.train_acc = metrics["train_acc"]
        self.labeltype = metrics["current_label"]
        self.c_epoch = metrics["current_epoch"]
        self.train_f1 = metrics["train_f1"]
        self.eval_loss = metrics["eval_loss"]
        self.eval_acc = metrics["eval_acc"]
        self.eval_f1 = metrics["eval_f1"]
        self.layer_freezing = metrics["frozen_until"]
        # Reconstruct the weights to match the previous state
        # Make sure class head output is the right size
        self.class_head[8] = nn.Linear(128, self.class_values[self.labeltype]).to(device)

        # Freeze weights based on previously defined stats
        until = self.layer_freezing[self.labeltype]
        if until:
            self.freezeencoders(until=until)

        # Make sure optim is the right size
        self.update_optimizer()

        # Load parameters
        self.load(f"{path}/latest.pt")

    
    def fit(self, train_dataset, final: bool = False):
        """Fit the model on a shuffled dataset for one epoch."""
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
            labels = batch["labels"]

            # Calculate the loss
            logits = self.forward(batch=batch)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward() 
            self.optimizer.step()

            # Save the loss
            train_loss.append(loss.item())

            # Only calculate metrics for the final epoch
            if final:
                # Compute the training metrics
                prediction = torch.argmax(logits, dim=1)
                train_correct += (prediction == labels).sum().item()
                train_total += labels.size(0)
                prediction_list.extend(prediction.cpu().detach().numpy())
                labels_list.extend(labels.cpu().detach().numpy())

            # Break the training loop when the set number of training steps are reached
            if timestep == self.steps_per_epoch - 1:
                break
                
            # Reset the previous index of the dataset
            prev = idx
        
        if final:
            return train_correct, train_total, train_loss, prediction_list, labels_list
        else:
            return train_loss
    

    def evaluate(self, eval_dataset, final: bool = True):
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
                labels = batch["labels"]

                # Calculate the loss
                logits = self.forward(batch=batch)
                loss = self.criterion(logits, labels)

                # Save the loss
                eval_loss.append(loss.item())

                # Only calculate metrics for the final epoch
                if final:
                    # Compute the validation metrics
                    prediction = torch.argmax(logits, dim=1)
                    eval_correct += (prediction == labels).sum().item()
                    eval_total += labels.size(0)
                    prediction_list.extend(prediction.cpu().detach().numpy())
                    labels_list.extend(labels.cpu().detach().numpy())

                # Break the validation loop when the set number of training steps are reached
                if timestep == self.steps_per_epoch - 1:
                    break
                    
                # Reset the previous index of the dataset
                prev = idx

        if final:
            return eval_correct, eval_total, eval_loss, prediction_list, labels_list
        
        else:
            return eval_loss


