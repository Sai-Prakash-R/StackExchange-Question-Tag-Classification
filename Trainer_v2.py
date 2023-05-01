"""
This is a multi-line docstring that can be used to provide an overview of the code file.

You can use this section to explain the purpose of the code, describe the different modules or classes that are imported,
provide examples of how the code can be used, or add any other relevant information.

Author: Sai Prakash Ravilisetty 
Date: 04/09/2023
"""
"""
This script defines a PyTorch metric class and imports several modules for deep learning and data manipulation.

Modules:
- pathlib: Object-oriented interface to the file system.
- Metric (from torchmetrics): Base class for defining custom metrics in PyTorch.
- torch: Main module for creating and training deep learning models in PyTorch.
- datetime: Classes for working with dates and times.
- matplotlib.pyplot: MATLAB-like plotting framework for creating visualizations.
- nn (from torch): Predefined layers and functions for building neural networks in PyTorch.
- random: Functions for generating random numbers and sequences.
- numpy: Support for large, multi-dimensional arrays and matrices, along with a large library of mathematical functions.
"""

from pathlib import Path
from torchmetrics import Metric
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import numpy as np



class Trainer:
    """
    A class for training deep learning models.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        criterion (callable): The loss criterion to use for training.
        device (str): The device to use for training.

    Attributes:
        train_loader (torch.utils.data.DataLoader): The data loader for training data.
        val_loader (torch.utils.data.DataLoader): The data loader for validation data.
        train_metric (callable): The evaluation metric to use for training.
        val_metric (callable): The evaluation metric to use for validation.
        early_stopping_step (int): The number of epochs to wait before stopping training if the validation score does not improve.
        save_best (bool): Whether to save the best model during training.
        save_every_n_epochs (int): The number of epochs between saving checkpoints.
        save_last_epoch (bool): Whether to save the model at the end of training.
        timestamp (str): The timestamp for the current training session.
        clipping (float): The gradient clipping threshold to use during training.
        best_score (float): The best validation score achieved so far.
        total_epochs (int): The total number of epochs trained so far.
        total_train_steps (int): The total number of training steps taken so far.
        total_val_steps (int): The total number of validation steps taken so far.
        best_epoch (int): The epoch with the best validation score.
        train_losses (list): A list of training losses.
        val_losses (list): A list of validation losses.
        train_metrics (list): A list of training evaluation metrics.
        val_metrics (list): A list of validation evaluation metrics.
        early_stopping_counter (int): A counter for the number of epochs since the last improvement in validation score.
        early_stop (bool): Whether to stop training early due to lack of improvement in validation score.

    Methods:
        set_loader(train_loader, val_loader): Sets the data loaders for training and validation data.
        set_metric(train_metric, val_metric): Sets the evaluation metrics for training and validation.
        set_early_stopping(early_stopping_step): Sets the early stopping criteria.
        set_checkpoint(save_best, save_every_n_epochs, save_last_epoch): Sets the checkpointing options.
        set_gradient_clipping(clipping): Sets the gradient clipping threshold.
        train(): Trains the model using the specified options.
    """
    def __init__(self, model, optimizer, criterion, device):
        """
        Initializes the Trainer class.

        Args:
            model (torch.nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            criterion (callable): The loss criterion to use for training.
            device (str): The device to use for training.
        """

        # input to constructor
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(self.device)

        # set using function set loader
        self.train_loader = None
        self.val_loader = None

        # set using function set metric
        self.train_metric = None
        self.val_metric = None

        # set using function set early stopping
        self.early_stopping_step = None

        # set using set checkpoint function
        self.save_best = None
        self.save_every_n_epochs = None
        self.save_last_epoch = None
        self.timestamp = None

        # set using set gradient clipping function
        self.clipping = None

        # updated during training
        self.best_score = None
        self.total_epochs = 0
        self.total_train_steps = 0
        self.total_val_steps = 0
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.early_stopping_counter = 0
        self.early_stop = False
        # self.learning_rates = []

    def set_early_stopping(self, patience=5, delta=0):
        """
            Sets up early stopping functionality for the model.

            Args:
            - patience (int): Number of epochs to wait before stopping training if validation loss doesn't improve.
            - delta (float): Minimum change in the monitored quantity to qualify as an improvement.

            Returns: None
        """
        def early_stopping_step(val_loss):
            """
            A helper function to check if early stopping criteria are met.

            Args:
            - val_loss (float): The validation loss for the current epoch.

            Returns: None
            """
            if self.best_score is not None:
                if val_loss >= self.best_score - delta:
                    self.early_stopping_counter += 1
                    print(
                        f'EarlyStopping counter: {self.early_stopping_counter} out of {patience}')
                    if self.early_stopping_counter >= patience:
                        self.early_stop = True
                else:
                    self.early_stopping_counter = 0
        self.early_stopping_step = early_stopping_step

    def set_loaders(self, train_loader, val_loader=None):
        """
        Sets the train and validation data loaders for the model.

        Args:
        - train_loader (DataLoader): The data loader for the training set.
        - val_loader (DataLoader): The data loader for the validation set. Defaults to None.

        Returns: None
        """
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_metric(self, train_metric: Metric, val_metric: Metric = None):
        """
        Sets the training and validation metrics for the model.

        Args:
        - train_metric (Metric): The metric to be used to evaluate the training set.
        - val_metric (Metric): The metric to be used to evaluate the validation set. Defaults to None.

        Returns: None
        """
        self.train_metric = train_metric
        self.val_metric = val_metric

    def set_checkpoint(self, save_path, save_best=True, save_every_n_epochs=None, save_last_epoch=False):
        """
        Sets up checkpoint saving functionality for the model.

        Args:
        - save_path (str): The path where the model checkpoints will be saved.
        - save_best (bool): If True, only save the best model according to validation loss. Defaults to True.
        - save_every_n_epochs (int): If provided, save the model every n epochs. Defaults to None.
        - save_last_epoch (bool): If True, save the model at the last epoch regardless of other conditions. Defaults to False.

        Returns: None
        """
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_path = save_path
        self.save_best = save_best
        self.save_every_n_epochs = save_every_n_epochs
        self.save_last_epoch = save_last_epoch

    def set_gradient_clipping(self, clip_type, clip_value, norm_type=2):
        """
        Sets up gradient clipping functionality for the model.

        Args:
        - clip_type (str): The type of gradient clipping to use. Either 'value' or 'norm'.
        - clip_value (float): The value to clip the gradients to.
        - norm_type (int): The type of norm to use if `clip_type` is 'norm'. Defaults to 2.

        Returns: None
        """

        if clip_type.lower() == 'value':
            self.clipping = lambda: nn.utils.clip_grad_value_(
                self.model.parameters(), clip_value=clip_value
            )
        elif clip_type.lower() == 'norm':
            self.clipping = lambda: nn.utils.clip_grad_norm_(
                self.model.parameters(), clip_value, norm_type
            )
        else:
            raise ValueError(
                "Invalid clip_type provided. Use 'value' or 'norm'.")

    def _get_batch_metric(self, outputs, targets, multilabel, metric, threshold):
        """
        Computes the batch-wise value of a given metric function for the current batch of data.

        Args:
        - outputs (torch.Tensor): The model's output for the current batch of data.
        - targets (torch.Tensor): The ground-truth labels for the current batch of data.
        - multilabel (bool): A boolean indicating whether the problem is multilabel or not.
        - metric (callable): The metric function to use for evaluating the model's performance.
        - threshold (float): The threshold value to use for multilabel classification problems.

        Returns:
        The value of the metric for the current batch of data.
        """

        if multilabel:
            outputs = (torch.sigmoid(outputs) > threshold).float()
        else:
            outputs = torch.argmax(outputs, dim=1)
        batch_metric = metric(outputs, targets)
        return batch_metric

    def _backward(self, loss, ):
         """
        Computes the gradients of the loss with respect to the model's parameters, applies gradient clipping
        (if specified), performs a parameter update using the optimizer, and resets the gradients to zero.

        Args:
        - loss (torch.Tensor): The loss value for the current batch of data.

        Returns:
        None
        """
        loss.backward()
        if callable(self.clipping):
            self.clipping()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _move_inputs_targets_to_gpu(self, inputs, targets):
        """
        Move the input and target tensors to the device (GPU or CPU) specified during trainer initialization.

        Args:
        - inputs: The input tensor(s) for the current batch of data.
        - targets: The target tensor(s) for the current batch of data.

        Returns:
        - inputs: The input tensor(s) moved to the specified device.
        - targets: The target tensor(s) moved to the specified device.
        """
        if isinstance(inputs, tuple):
            inputs = tuple(input_tensor.to(self.device)
                           for input_tensor in inputs)
        else:
            inputs = inputs.to(self.device)

        targets = targets.to(self.device)

        return inputs, targets

    def _log_print_epoch_loss_metric(self, train_loss, train_metric, val_loss, val_metric, epoch,
                                     num_epochs, dt_train, dt_valid):
        """
        Print the training and validation loss/metric and time taken for each epoch during training.

        Args:
            train_loss (float): the training loss for the current epoch
            train_metric (float): the training metric for the current epoch
            val_loss (float): the validation loss for the current epoch
            val_metric (float): the validation metric for the current epoch
            epoch (int): the current epoch number
            num_epochs (int): the total number of epochs
            dt_train (float): the time taken for training the current epoch
            dt_valid (float): the time taken for validating the current epoch

        Returns:
            None
        """
        if self.train_metric:
            print(
                f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
                f"Train Metric: {train_metric:.4f}, Train Time: {dt_train}")

        else:
            print(
                f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {val_loss:.4f}, Train Time: {dt_train}")

        if self.val_loader is not None:
            if self.val_metric:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}, "
                    f"Val Metric: {val_metric:.4f}, Val Time: {dt_valid}")

            else:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}, Val Time: {dt_valid}")
        # print(f"Current Learning rate is {self.learning_rates[-1]}")
        print()

    def _run_batch(self, inputs, targets, metric, multilabel, threshold, training):
        """
        Runs a single batch of inputs through the model and computes the loss and/or metric.

        Args:
            inputs (torch.Tensor): the input tensor for the batch
            targets (torch.Tensor): the target tensor for the batch
            metric (callable): a metric function to compute on the batch, defaults to None
            multilabel (bool): whether the targets are multilabel or not, defaults to False
            threshold (float): the threshold to use for multilabel classification, defaults to 0.5
            training (bool): whether the model is in training mode or not, defaults to True

        Returns:
            loss (torch.Tensor): the loss tensor for the batch
        """

        inputs, targets = self._move_inputs_targets_to_gpu(
            inputs, targets)

        with torch.set_grad_enabled(training):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            if training:
                self._backward(loss, )

        if metric is not None:
            batch_metric = self._get_batch_metric(
                outputs, targets, multilabel, metric, threshold)

        if training:
            self.total_train_steps += 1
        else:
            self.total_val_steps += 1

        return loss

    def _run_epoch(self, loader, training=True, multilabel=False, threshold=0.5):
        """
        Runs a single epoch of the data loader and computes the epoch loss and/or metric.

        Args:
            loader (torch.utils.data.DataLoader): the data loader for the epoch
            training (bool): whether the model is in training mode or not, defaults to True
            multilabel (bool): whether the targets are multilabel or not, defaults to False
            threshold (float): the threshold to use for multilabel classification, defaults to 0.5

        Returns:
            epoch_loss (float): the average loss over the epoch
            epoch_metric (float or None): the average metric over the epoch, or None if no metric was computed
        """

        if training:
            self.model.train()
            metric = self.train_metric
        else:
            self.model.eval()
            metric = self.val_metric

        epoch_loss = 0.0
        num_samples = 0

        for i, (inputs, targets) in enumerate(loader):
            loss = self._run_batch(
                inputs, targets, metric, multilabel, threshold, training)

            epoch_loss += loss.item() * targets.size(0)
            num_samples += targets.size(0)

        epoch_loss /= num_samples

        if metric is not None:
            epoch_metric = metric.compute().item()
            metric.reset()
        else:
            epoch_metric = None

        return epoch_loss, epoch_metric

    def save_checkpoint(self, suffix=''):
        """
        Saves a checkpoint of the model and optimizer state, as well as the loss and metric histories.

        Args:
            suffix (str): an optional suffix to append to the checkpoint filename, defaults to ''

        Returns:
            None
        """
        save_dir = Path(self.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = save_dir / f"checkpoint_{self.timestamp}{suffix}.pt"
        checkpoint_data = {
            'total_epochs': self.total_epochs,
            'total_train_steps': self.total_train_steps,
            'total_val_steps': self.total_val_steps,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.best_score,
        }

        # Add losses and metrics history to the checkpoint
        checkpoint_data['train_losses'] = self.train_losses
        checkpoint_data['val_losses'] = self.val_losses
        checkpoint_data['train_metrics'] = self.train_metrics
        checkpoint_data['val_metrics'] = self.val_metrics

        torch.save(checkpoint_data, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        """
        Loads a saved checkpoint of a model training process from a file path, and restores the saved model weights,
        optimizer state, and other training-related information.

        Args:
            checkpoint_path (str): The file path of the saved checkpoint.

        Returns:
            None
        """

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_epochs = checkpoint['total_epochs']
        self.total_train_steps = checkpoint['total_train_steps']

        if 'total_val_steps' in checkpoint:
            self.total_val_steps = checkpoint['total_val_steps']
        if 'val_loss' in checkpoint:
            self.best_score = checkpoint['val_loss']
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'train_metrics' in checkpoint:
            self.train_metrics = checkpoint['train_metrics']
        if 'val_metrics' in checkpoint:
            self.val_metrics = checkpoint['val_metrics']

        print(f"Loaded checkpoint from '{checkpoint_path}'.")

    def train(self, num_epochs, multilabel=False, threshold=0.5):
        """
        Trains a machine learning model using a given dataset for a specified number of epochs.

        Args:
            num_epochs (int): The number of epochs to train the model.
            multilabel (bool, optional): Whether the model is for multi-label classification or not. Defaults to False.
            threshold (float, optional): The threshold value for the model's predicted output. Defaults to 0.5.

        Returns:
            None
        """
        assert self.train_loader is not None, "Train loader must be set before calling train()"

        if self.val_loader is None:
            print(
                'Validation loader is not set. The trainer will only execute training Loop')

        if all(value is None for value in [self.save_best, self.save_every_n_epochs, self.save_last_epoch]):
            print('Not saving any checkpoint')

        for epoch in range(num_epochs):

            t0 = datetime.now()

            train_loss, train_metric = self._run_epoch(
                self.train_loader, training=True, multilabel=multilabel, threshold=threshold)

            dt_train = datetime.now() - t0

            self.train_losses.append(train_loss)

            if self.train_metric:
                self.train_metrics.append(train_metric)

            if self.val_loader is not None:
                t0 = datetime.now()
                val_loss, val_metric = self._run_epoch(
                    self.val_loader, training=False, multilabel=multilabel,
                    threshold=threshold)
                dt_valid = datetime.now() - t0

                self.val_losses.append(val_loss)

                if self.val_metric:
                    self.val_metrics.append(val_metric)

                if callable(self.early_stopping_step):
                    self.early_stopping_step(val_loss)
                    if self.early_stop:
                        print("Early stopping triggered")
                        break

                if self.best_score is None or val_loss < self.best_score:
                    self.best_score = val_loss
                    self.best_epoch = self.total_epochs + 1
                    if self.save_best:
                        self.save_checkpoint(suffix=f'_best')

            # saving checkpoint

            if self.save_every_n_epochs and (epoch + 1) % self.save_every_n_epochs == 0:
                self.save_checkpoint(
                    suffix=f'_epoch_{self.total_epochs + 1}')

            if self.save_last_epoch:
                self.save_checkpoint(
                    suffix=f'_last')

            self._log_print_epoch_loss_metric(train_loss, train_metric, val_loss, val_metric, epoch,
                                              num_epochs, dt_train, dt_valid)

            self.total_epochs += 1

    def plot_history(self):

        """
        Plots the training and validation loss and (if available) metric as a function of the number of epochs.

        Parameters:
        - None

        Returns:
        - None
        """

        epochs = range(1, len(self.train_losses) + 1)

        plt.figure()
        plt.plot(epochs, self.train_losses, label="Train")
        plt.plot(epochs, self.val_losses, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        if self.train_metrics[0] is not None:
            plt.figure()
            plt.plot(epochs, self.train_metrics, label="Train")
            plt.plot(epochs, self.val_metrics, label="Validation")
            plt.xlabel("Epochs")
            plt.ylabel("Metric")
            plt.legend()
            plt.show()

    def predict(self, loader, return_targets=False, multilabel=False, threshold=0.5):
        """
        Computes model predictions on a given data loader.

        Parameters:
        - loader: A PyTorch DataLoader object representing the data to make predictions on.
        - return_targets: A boolean indicating whether to return the ground-truth targets alongside the predicted targets.
        - multilabel: A boolean indicating whether the model is trained for multilabel classification.
        - threshold: A float threshold value used for multilabel classification, representing the probability above which a label is considered positive.

        Returns:
        - If return_targets is True, a tuple containing two elements:
            - A PyTorch tensor of shape (num_samples,) containing the predicted targets.
            - A PyTorch tensor of shape (num_samples,) containing the ground-truth targets.
        - If return_targets is False, a PyTorch tensor of shape (num_samples,) containing the predicted targets.
        """

        self.model.to(self.device)
        self.model.eval()

        predictions = []
        targets_list = []

        with torch.no_grad():
            for inputs, targets in loader:

                inputs, targets = self._move_inputs_targets_to_gpu(
                    inputs, targets)

                outputs = self.model(inputs)

                if multilabel:
                    outputs = (torch.sigmoid(outputs) > threshold).float()
                else:
                    outputs = torch.argmax(outputs, dim=1)

                predictions.append(outputs.cpu())
                targets_list.append(targets.cpu())

        predictions_tensor = torch.cat(predictions, dim=0)
        targets_tensor = torch.cat(targets_list, dim=0)

        if return_targets:
            return predictions_tensor, targets_tensor,
        else:
            return predictions_tensor

    def sanity_check(self, num_classes):
        """
        The sanity_check method of the Trainer class checks if the model is initialized correctly by verifying that the initial loss value is close to the theoretical loss value, given the number of classes in the classification problem.

        Args:
        - num_classes (int): The number of classes in the classification problem.

        Returns:
        None

        Prints:
        - Actual loss: The actual loss value calculated by the model on the first batch of the train loader.
        - Expected Theoretical loss: The theoretical loss value calculated based on the number of classes in the classification problem.

        Note:
        - This method assumes that the model has already been initialized and the train_loader has been set.
        """
        for inputs, targets in self.train_loader:

            inputs, targets = self._move_inputs_targets_to_gpu(
                inputs, targets)

            self.model.eval()
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            print(f'Actual loss: {loss}')
            break
        print(f'Expected Theoretical loss: {np.log(num_classes)}')
        self.model.train()

    @staticmethod
    def set_seed(seed=42):
        """
        Set seed function for reproducibility.

        Args:
        seed (int): Seed value to set for PyTorch, Numpy and Python's built-in random module.
        Default is 42.

        Returns:
        None
        """
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
