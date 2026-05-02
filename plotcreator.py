import matplotlib.pyplot as plt
import matplotlib
from glob import glob
import sys
import numpy as np

options = sys.argv[1:]

colours = ["b", "g", "r", "m"]
if __name__ == "__main__":
    matplotlib.rc('font', size=9)
    fig, ax = plt.subplots()
    name = options[0]
    for i, file in enumerate(glob(f"{options[0]}/*")):
        metrics = np.load(file, allow_pickle=True).item()
        # This should probably work
        run_name = file.split(".npy")[-2].split("/")[-1]
        # Load loss etc
        epochs = metrics["epochs"]["species"]
        # Get epoch range value properly
        train_loss = metrics["train_loss"].mean(axis=1)[:-1]
        eval_loss = metrics["eval_loss"].mean(axis=1)[:-1]
        print(f"{run_name}\nTrain_acc: {metrics["train_acc"]}\nTrain_F1: {metrics["train_f1"]}\nEval_acc: {metrics["eval_acc"]}\nEval_F1: {metrics["eval_f1"]}")
        # Make plot, 2 lines of same colour
        # Train loss is dotted line
        epoch_range = [x for x in range (1, epochs + 1)]
        ax.plot(epoch_range, train_loss[0:epochs], linestyle='--', label=f"{run_name}_train", color=colours[i])
        # Eval loss is solid line
        ax.plot(epoch_range, eval_loss[0:epochs], linestyle="-", label=f"{run_name}_val", color=colours[i])

    ax.set(xlabel="Epochs", ylabel="Loss", title=f"{name} Traning and validation loss")
    fig.set_size_inches(6, 4)
    plt.yticks([x for x in range(int(plt.yticks()[0][0]), int(plt.yticks()[0][-1] + 4), 2)])
    plt.legend(prop={'size': 6})
    plt.savefig(f"{name}_lossplot.png", dpi=300)
