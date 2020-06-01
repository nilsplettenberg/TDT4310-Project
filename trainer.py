import pathlib
import matplotlib.pyplot as plt
import torch
import utils
import time
import typing
import collections
import numpy as np
from torch import nn

def compute_precision(tp, tn, fp,fn):
    """
        Returns:
            (precision, recall, accuracy)
    """
    for idx, c in enumerate(["bot", "female", "male"]):          
        try:
            p = tp[idx]/(tp[idx]+fp[idx])
        except ZeroDivisionError:
            p = 0
        try:
            r = tp[idx]/(tp[idx]+fn[idx]) 
        except ZeroDivisionError:
            r = 0
        try:
            a = (tp[idx]+tn[idx])/(tp[idx]+fp[idx]+tn[idx]+fn[idx])
        except ZeroDivisionError:
            a = 0
        print("Class %6s: precision: %.3f, recall:%.3f, accuracy:%.3f" % (c,p,r,a))
    return p, r, a

def compute_precision_users(user_stats: dict, threshold = 0.5):
    """
        Computes precision, recall and accuracy based on all tweets of a user
    """
    # true positives, false positives, true negatives, false negatives for all classes
    tp =[0,0,0]
    fp =[0,0,0]
    tn =[0,0,0]
    fn =[0,0,0]
    for user, stats in user_stats.items():
        p = np.argmax(stats[:-1])
        y = stats[-1] 
        confidence = max(stats[:-1])/sum(stats[:-1])
        if p == y:
            tp[p] += 1
            tn[p-1] += 1
            tn[p-2] += 1
        else:
            fn[y] += 1
            fp[p] += 1
            tn[-p-y] += 1
    a = sum(tp) / len(user_stats)
    print("Accuracy for users: %.3f" % a)
    compute_precision(tp, tn, fp, fn)

def compute_loss_and_accuracy(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_criterion: torch.nn.modules.loss._Loss):
    """
    Computes the average loss and the accuracy over the whole dataset
    in dataloader.
    Args:
        dataloder: Validation/Test dataloader
        model: torch.nn.Module
        loss_criterion: The loss criterion, e.g: torch.nn.CrossEntropyLoss()
    Returns:
        [average_loss, accuracy]: both scalar.
    """
    average_loss = 0
    accuracy = 0

    loss = []
    correct = 0
    total = 0
    num_classes = model.num_classes

    with torch.no_grad():
        # true positives, false positives, true negatives, false negatives for all classes
        tp =[0,0,0]
        fp =[0,0,0]
        tn =[0,0,0]
        fn =[0,0,0]
        # save stats per user
        user_stats = {}
        for (X_batch, Y_batch) in dataloader:
            # Transfer images/labels to GPU VRAM, if possible
            X_batch = utils.to_cuda(X_batch)
            Y_batch = utils.to_cuda(Y_batch)
            # Forward pass the images through our model
            output_probs = model(X_batch)
            # print(output_probs[1])

            # Compute Loss and Accuracy
            loss.append(loss_criterion(output_probs, Y_batch).item())

            _, predicted = torch.max(output_probs.data, 1)
            total += Y_batch.size(0)
            correct += (predicted == Y_batch).sum().item()

            user_ids = X_batch[:,-1]
            #compute precision, recall a accuracy per class
            # only works for 3 classes
            for idx, prediction in enumerate(predicted):
                p = int(prediction)
                y = int(Y_batch[idx])
                user_id = int(torch.mean(user_ids[idx]))
                if p == y:
                    tp[p] += 1
                    tn[p-1] += 1
                    tn[p-2] += 1
                else:
                    fn[y] += 1
                    fp[p] += 1
                    tn[-p-y] += 1
                try:
                    user_stats[user_id][prediction] +=1
                except KeyError:
                    user_stats[user_id] = [0,0,0,y] 
                    user_stats[user_id][prediction] +=1
    compute_precision(tp, tn, fp, fn)
    compute_precision_users(user_stats)
    average_loss = np.mean(loss)
    accuracy = 100 * correct / total
    return average_loss, accuracy



class Trainer:

    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 early_stop_count: int,
                 epochs: int,
                 model: torch.nn.Module,
                 datasets,
                 loss_criterion: torch.nn.modules.loss._Loss = nn.CrossEntropyLoss(),
                 optimizer= None):
        """
            Initialize our trainer class.
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop_count = early_stop_count
        self.epochs = epochs

        # Since we are doing multi-class classification, we use CrossEntropyLoss
        self.loss_criterion = loss_criterion
        # Initialize the model
        self.model = model
        # Transfer model to GPU VRAM, if possible.
        self.model = utils.to_cuda(self.model)
        print(self.model)

        # Define our optimizer. SGD = Stochastich Gradient Descent
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(),self.learning_rate)
        #self.optimizer = torch.optim.Adam(model.parameters(), self.learning_rate, weight_decay=0.0001)
        

        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = datasets

        # Validate our model everytime we pass through 50% of the dataset
        self.num_steps_per_val = len(self.dataloader_train) // 2
        self.global_step = 0
        self.start_time = time.time()

        # Tracking variables
        self.VALIDATION_LOSS = collections.OrderedDict()
        self.TEST_LOSS = collections.OrderedDict()
        self.TRAIN_LOSS = collections.OrderedDict()
        self.VALIDATION_ACC = collections.OrderedDict()
        self.TEST_ACC = collections.OrderedDict()

        self.checkpoint_dir = pathlib.Path("checkpoints")

    def validation_epoch(self):
        """
            Computes the loss/accuracy for all three datasets.
            Train, validation and test.
        """
        self.model.eval()
        validation_loss, validation_acc = compute_loss_and_accuracy(
            self.dataloader_val, self.model, self.loss_criterion
        )
        self.VALIDATION_ACC[self.global_step] = validation_acc
        self.VALIDATION_LOSS[self.global_step] = validation_loss
        used_time = time.time() - self.start_time
        print(
            f"Epoch: {self.epoch:>2}",
            f"Batches per seconds: {self.global_step / used_time:.2f}",
            f"Global step: {self.global_step:>6}",
            f"Validation Loss: {validation_loss:.2f},",
            f"Validation Accuracy: {validation_acc:.3f}",
            sep="\t")
        # Compute for testing set
        test_loss, test_acc = compute_loss_and_accuracy(
            self.dataloader_test, self.model, self.loss_criterion
        )
        self.TEST_ACC[self.global_step] = test_acc
        self.TEST_LOSS[self.global_step] = test_loss

        self.model.train()

    def should_early_stop(self):
        """
            Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        if len(self.VALIDATION_LOSS) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = list(self.VALIDATION_LOSS.values())[-self.early_stop_count:]
        first_loss = relevant_loss[0]
        if first_loss == min(relevant_loss):
            print("Early stop criteria met")
            return True
        return False

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """
        # Track initial loss/accuracy
        def should_validate_model():
            return self.global_step % self.num_steps_per_val == 0

        for epoch in range(self.epochs):
            self.epoch = epoch
            # Perform a full pass through all the training samples
            for X_batch, Y_batch in self.dataloader_train:
                # Transfer images / labels to GPU VRAM, if possible
                X_batch = utils.to_cuda(X_batch)
                Y_batch = utils.to_cuda(Y_batch)

                # Perform the forward pass
                predictions = self.model(X_batch)
                # Compute the cross entropy loss for the batch
                loss = self.loss_criterion(predictions, Y_batch)
                self.TRAIN_LOSS[self.global_step] = loss.detach().cpu().item()

                # Backpropagation
                loss.backward()

                # Gradient descent step
                self.optimizer.step()

                # Reset all computed gradients to 0
                self.optimizer.zero_grad()
                self.global_step += 1
                 # Compute loss/accuracy for all three datasets.
                if should_validate_model():
                    self.validation_epoch()
                    self.save_model()
                    if self.should_early_stop():
                        print("Early stopping.")
                        return

    def save_model(self):
        def is_best_model():
            """
                Returns True if current model has the lowest validation loss
            """
            validation_losses = list(self.VALIDATION_LOSS.values())
            return validation_losses[-1] == min(validation_losses)

        state_dict = self.model.state_dict()
        filepath = self.checkpoint_dir.joinpath(f"{self.global_step}.ckpt")

        utils.save_checkpoint(state_dict, filepath, is_best_model())

    def load_best_model(self):
        state_dict = utils.load_best_checkpoint(self.checkpoint_dir)
        if state_dict is None:
            print(
                f"Could not load best checkpoint. Did not find under: {self.checkpoint_dir}")
            return
        self.model.load_state_dict(state_dict)
        
    def report_final_loss(self):
        self.load_best_model()
        train_loss, train_acc = compute_loss_and_accuracy(
            self.dataloader_train, self.model, self.loss_criterion
        )
        validation_loss, validation_acc = compute_loss_and_accuracy(
            self.dataloader_val, self.model, self.loss_criterion
        )
        test_loss, test_acc = compute_loss_and_accuracy(
            self.dataloader_test, self.model, self.loss_criterion
        )
        print(f"Final Training Loss: {train_loss:.2f}", f"Final Training accuracy: {train_acc:.3f}", sep="\t")
        print(f"Final Validation Loss: {validation_loss:.2f}", f"Final Validation accuracy: {validation_acc:.3f}", sep="\t")
        print(f"Final Test Loss: {test_loss:.2f}", f"Final Test accuracy: {test_acc:.3f}", sep="\t")

def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.TRAIN_LOSS, label="Training loss")
    utils.plot_loss(trainer.VALIDATION_LOSS, label="Validation loss")
    utils.plot_loss(trainer.TEST_LOSS, label="Testing Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.VALIDATION_ACC, label="Validation Accuracy")
    utils.plot_loss(trainer.TEST_ACC, label="Testing Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()