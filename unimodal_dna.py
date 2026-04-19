import torch
import os
import sys
import numpy as np
import random
from transformers import AutoTokenizer, AutoModel, set_seed
from transformers import get_linear_schedule_with_warmup
from torch import nn
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
apitoken = os.getenv("API_KEY")

global device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DNAEncoder(nn.Module):
    def __init__(
            self,
            params: dict,
            run_name: str,
            d_enc: str = "zhihan1996/DNA_bert_6",
            d_tokenizer: str = "zhihan1996/DNA_bert_6",
            cache_dir: str | bool = False,
            ds_randomization: bool = False,
            augmentation: bool = False,
            hierarchical: bool = False,
        ):
        super(DNAEncoder, self).__init__()
        
        # Load pretrained DNA-BERT Model
        if cache_dir:
            self.dna_encoder = AutoModel.from_pretrained(
                d_enc,
                token=apitoken,
                cache_dir=cache_dir,
            )
            self.dna_tokenizer = AutoTokenizer.from_pretrained(
                d_tokenizer,
                token=apitoken,
                cache_dir=cache_dir,
            )
        else:
            self.dna_encoder = AutoModel.from_pretrained(
                d_enc,
                token=apitoken,
            )
            self.dna_tokenizer = AutoTokenizer.from_pretrained(
                d_tokenizer,
                token=apitoken,
            )
        
        # init parameters
        self.lr = params["lr"]

        self.class_values = params["class_values"]
        self.class_mapping = params["class_mapping"]
        self.epoch_ordering = params["epoch_ordering"]
        self.layer_freezing = params["layer_freezing"]

        self.steps_per_epoch = params["steps_per_epoch"]
        self.batch_size = params["batch_size"]
        self.k = params["k"]

        self.hierarchical = hierarchical
        self.ds_rand = ds_randomization
        self.augmentation = augmentation

        total_epochs = sum(self.epoch_ordering.values())
        self.run_name = run_name
        
        # Hardcoded output sizes of DINOV2_small and DNABERT_6
        d_enc_size = 768

        # Create projection for to make BERT features better suited for classification
        self.class_head = nn.Sequential(
            nn.Linear(d_enc_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128), 
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, [i for i in self.class_values.values()][0])
            ) # Maps embedding to species classes
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        # Training metrics storage
        self.train_loss = np.zeros(shape=[total_epochs + 1, self.steps_per_epoch])
        self.eval_loss = np.zeros(shape=[total_epochs + 1, self.steps_per_epoch])
        self.train_acc = 0
        self.eval_acc = 0
        self.train_f1 = 0
        self.eval_f1 = 0
        
        # If the weights exist, we have to run from the checkpoint
        if os.path.exists(f"./{run_name}/latest.pt"):
            self.start_from_checkpoint(f"./{run_name}/")


    def collate_fn(self, batch):
        """Custom collation function for dataloader, extract images from the batch and pad them with the largest width from the widest image.

        Args:
            batch(dict): The batch containing the data subset

        Returns:
            torch.tensor: The stacked tensor of the batch of images.
        """

        labels = [self.class_mapping[self.labeltype][i] for i in batch[self.labeltype]]
        
        if self.augmentation:
            barcodes = self.kmer_crop(batch["dna_barcode"])
        else:
            barcodes = [" ".join([seq[i:i+self.k] for i in range(len(seq) - self.k + 1)]) for seq in batch["dna_barcode"]]


        return {"labels": torch.tensor(labels).long().to(device),
                "barcodes": barcodes}


    def kmer_crop(self, barcodes, max_length=510, center: bool = False):    
        """Crop a random portion of kmers from the barcode.
        
        Args:
            barcodes (list): List of DNA barcodes
            max_length (integer): Maximum kmers to output, standard 510.
        
        Returns:
            List of kmer crop kmers.
        """
        if center:
            starts = [(len(sequence) - max_length) // 2 for sequence in barcodes]            
        else:
            starts = [np.random.randint(0, len(sequence) - max_length) if len(sequence) > max_length - 1 else 0 for sequence in barcodes]
        
        crops = [seq[start:start + max_length] for start, seq in zip(starts, barcodes)]
        kmer_crops = [" ".join([seq[i:i+self.k] for i in range(len(seq) - self.k + 1)]) for seq in crops]
        return kmer_crops
        

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
            self.freeze_until(self.dna_encoder, until)

        # Make sure optim is the right size
        self.update_optimizer()

        # Load parameters
        self.load(f"{path}/latest.pt")


    def encode(self, input_ids, attention_mask):
        outputs = self.dna_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Mean Pooling
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()

        embedding = (hidden * mask).sum(dim=1)
        embedding = embedding / mask.sum(dim=1).clamp(min=1e-6)

        return embedding


    def forward(self, dna):
        # CLS token as embedding
        embedding = self.dna_encoder(**dna).last_hidden_state[:,0]
        logits = self.class_head(embedding)

        return logits

    def evaluate(self, eval_dataset, final: bool = False):
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
                barcodes = batch["barcodes"]
                tokenized_barcodes = self.dna_tokenizer(barcodes, return_tensors = 'pt', padding=True, truncation=True).to(device)
                
                # Loss calculation
                logits = self.forward(dna=tokenized_barcodes)
                loss = self.criterion(logits, labels)
                
                # Save the loss
                eval_loss.append(loss.item())

                # Compute these metrics only on the final epoch
                if final:
                    # Compute metrics
                    prediction = torch.argmax(logits, dim=1)
                    eval_correct += (prediction == labels).sum().item()
                    eval_total += labels.size(0)
                    prediction_list.extend(prediction.cpu().detach().numpy())
                    labels_list.extend(labels.cpu().detach().numpy())

                # Break the loop when the set number of training steps are reached
                if timestep == self.steps_per_epoch - 1:
                    break
                    
                # Reset prev
                prev = idx
        if final:
            return eval_correct, eval_total, eval_loss, prediction_list, labels_list
        else:
            return eval_loss


    def fit(self, train_dataset, final: bool = False):
        """Fit the model on a shuffled dataset for one epoch"""
        self.train() # Turn on the dropouts
        prev = 0   # Previous index of the dataset
        train_loss = [] # List for the training loss
        train_correct = 0 # The number of correct predictions
        train_total = 0 # Total number of training labels
        prediction_list = [] # List for all predictions done per epoch
        labels_list = [] # List for all labels per epoch

        for timestep, idx in enumerate(range(self.batch_size, len(train_dataset), self.batch_size)):
            self.optimizer.zero_grad() # Zero out previous grad

            # Collate data properly
            batch = self.collate_fn(train_dataset[prev:idx])

            # Single out data
            labels = batch["labels"]
            barcodes = batch["barcodes"]
            tokenized_barcodes = self.dna_tokenizer(barcodes, return_tensors = 'pt', padding=True, truncation=True).to(device)

            # Calculate loss
            logits = self.forward(dna=tokenized_barcodes)
            loss = self.criterion(logits, labels)
            
            # Backwards pass
            loss.backward()
            self.optimizer.step()

            # Track loss
            train_loss.append(loss.item())

            # To make the runtime faster, we only calculate metrics for the final epoch
            if final:
                # Compute metrics
                prediction = torch.argmax(logits, dim=1)
                train_correct += (prediction == labels).sum().item()
                train_total += labels.size(0)
                prediction_list.extend(prediction.cpu().detach().numpy())
                labels_list.extend(labels.cpu().detach().numpy())
            
            # Break training when steps are done
            if timestep == self.steps_per_epoch - 1:
                break
                
            # reset prev
            prev = idx
            
        if final:
            return train_correct, train_total, train_loss, prediction_list, labels_list
        else:
            return train_loss
    
    def train_loop(self, train_dataset, eval_dataset):
        # Loop through class label types
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
            # Current type of class/label
            self.labeltype = labeltype
            epochs = self.epoch_ordering[labeltype] # Get number of epochs for the label
            
            until = self.layer_freezing[self.labeltype] # How many layers to freeze
            if until: # If None, no freeze.
                # Freeze the bottom n layers of both encoder
                self.freeze_until(self.dna_encoder, until)
            
            # Make sure a checkpoint load doesnt destoy the old class head
            if not self.check_start:
                # Replace final class head layer with a new output layer matching the number of classes
                self.class_head[8] = nn.Linear(128, self.class_values[self.labeltype]).to(device)
                # Create epoch range for new epoch loop
                epoch_range = [i for i in range(epochs)]

                # Make sure optim only optimizes trainable weights with requires_grad=true
                self.update_optimizer()
            
            for epoch in epoch_range:
                self.c_epoch = epoch + 1
                # Shuffle dataset
                if self.ds_rand:
                    train_dataset = train_dataset.shuffle()
                print(f"Epoch {epoch+1}/{epochs} for {labeltype}.")

                # Check for final epoch when calling self.fit and evaluate.
                if self.c_epoch == epochs:
                    train_correct, train_total, train_loss, train_preds, train_labels = self.fit(train_dataset=train_dataset, final=True)
                    eval_correct, eval_total, eval_loss, eval_preds, eval_labels = self.evaluate(eval_dataset=eval_dataset, final=True)
                else:
                    train_loss = self.fit(train_dataset=train_dataset)
                    eval_loss = self.evaluate(eval_dataset=eval_dataset)
                
                # Save and print the metrics for the current epoch
                self.train_loss[epoch, :] = np.array(train_loss)
                print("Training loss: ", self.train_loss.mean(axis=1)[epoch])
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
        print(f"Training accuracy: {self.eval_acc} Training F1 score: {self.eval_f1}")
        print("Training loss log: ", self.train_loss.mean(axis=1)[:-1])


    def freeze_until(self, model, until):
        """Freezes weights of given model given the value. If until=1, first 2 layers are frozen."""
        for i, layer in enumerate(model.encoder.layer):
            # If the index is withing the to be frozen layers
            if i <= until:
                for param in layer.parameters():
                    param.requires_grad = False
    

    def update_optimizer(self):
        """Updates optimizer to only use trainable model params"""
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.lr)

    
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
                   "dataset_randomization": self.ds_rand,
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

    options = sys.argv[1:]
    run_name = options[0]
    hierarchical = True if "hierarchical" in options else False
    ds_randomization = True if "ds_rand" in options else False
    augmentation = True if "augment" in options else False
    
    # Create save location directory
    if not os.path.exists(f"./{run_name}/"):
        os.mkdir(f"./{run_name}/")
    
    # If cache needs to be used
    cache_dir = os.getenv("cache_dir") if os.path.isdir(os.getenv("cache_dir")) else None

    # Train dataset
    train_dataset = load_dataset(
        "dataset.py",
        name="cropped_256_train",
        split="train",
        trust_remote_code=True,
        token=apitoken,
        cache_dir=cache_dir
    )
    train_dataset = train_dataset.with_format("torch", device=device)
    
    # Load the BIOSCAN5M validation dataset
    eval_dataset = load_dataset("dataset.py", 
                                name="cropped_256_eval", 
                                split="validation", 
                                trust_remote_code=True,
                                token=apitoken,
                                cache_dir=cache_dir
    )
    eval_dataset = eval_dataset.with_format("torch", device=device)
    
        # Initialize every single species as a valuen integer
    uniq_classes = set.union(set(train_dataset["class"]), set(train_dataset["class"]))
    class_dict = {entry: i for i, entry in enumerate(uniq_classes)}
    n_class = len(uniq_classes)

    uniq_orders = set.union(set(train_dataset["order"]), set(eval_dataset["order"]))
    order_dict = {entry: i for i, entry in enumerate(uniq_orders)}
    n_orders = len(uniq_orders)

    uniq_families = set.union(set(train_dataset["family"]), set(eval_dataset["family"]))
    family_dict = {entry: i for i, entry in enumerate(uniq_families)}
    n_family = len(uniq_families)

    uniq_genus = set.union(set(train_dataset["genus"]), set(eval_dataset["genus"]))
    genus_dict = {entry: i for i, entry in enumerate(uniq_genus)}
    n_genus = len(uniq_genus)

    uniq_species = set.union(set(train_dataset["species"]), set(train_dataset["species"]))
    species_dict = {entry: i for i, entry in enumerate(uniq_species)}
    n_species = len(uniq_species)
    
    # Set is fully non-deterministic, so we fix that by using mapping files to class indices
    if not os.path.exists("./class_indices/"):
        os.mkdir("./class_indices/")
        np.save("./class_indices/class.npy", class_dict)
        np.save("./class_indices/order.npy", order_dict)
        np.save("./class_indices/family.npy", family_dict)
        np.save("./class_indices/genus.npy", genus_dict)
        np.save("./class_indices/species.npy", species_dict)
    else:
        class_dict = np.load("./class_indices/class.npy", allow_pickle=True).item()
        order_dict = np.load("./class_indices/order.npy", allow_pickle=True).item()
        family_dict = np.load("./class_indices/family.npy", allow_pickle=True).item()
        genus_dict = np.load("./class_indices/genus.npy", allow_pickle=True).item()
        species_dict = np.load("./class_indices/species.npy", allow_pickle=True).item()

    if hierarchical:
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
                "family": 10,
                "genus": 25,
                "species": 100
                }, # Number of epochs for each step.
            layer_freezing = {
                "class": None,
                "order": 1,
                "family": 4,
                "genus": 5,
                "species": 9,
                },
            k=6,
        )
    else:
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
                "species": 100
                }, # Number of epochs for each step.
            layer_freezing = {
                "species": None,
                },
            k=6,
        )

    model = DNAEncoder(
        cache_dir=cache_dir,
        params=parameters,
        run_name = run_name,
        ds_randomization=ds_randomization,
        augmentation=augmentation,
        hierarchical=hierarchical,
    )

    model = model.to(device)
    model.train_loop(train_dataset, eval_dataset)
    model.save(f"./{run_name}/")
    model.plot_metrics(save_path=f"./{run_name}")
    