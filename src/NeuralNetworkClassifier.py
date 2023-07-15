import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import time
from .NeuralNetwork import NeuralNetwork

class NeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn wrapper for a neural network classifier.
    """
    def __init__(
        self,
        in_features=768, # NOTE: this is the number of features in BERT
        out_features=50,
        n_classes=2,
        epochs=10,
        learning_rate=0.01,
        weight_decay=0.0,
        momentum=0.0,
        batch_size=32,
        verbose=False,
        optimizer="adam"
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.n_classes = n_classes

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.batch_size = batch_size
        self.verbose = verbose
        self.classes_ = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.train_losses_ = []
        self.loss_ = None
        self.criterion = nn.CrossEntropyLoss()
        self.model = NeuralNetwork(
            self.in_features,
            self.out_features,
            self.n_classes
        )
        self.model.to(self.device)
        if optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        else:
            self.optimizer = optimizer
        self.optimizer.zero_grad()


    def fit(self, X, y):
        """
        Args:
            X (torch.Tensor): features set
            y (torch.Tensor): target label
        """
        X = X.to(self.device)
        y = y.to(self.device)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size
        )

        for epoch in range(self.epochs):
            start_time = time.time()
            batch_losses = []

            for _, (X_batch, y_batch) in enumerate(dataloader):
                loss = self._backprop(X_batch, y_batch)

                batch_losses.append(loss)

            self.train_losses_.append(np.mean(batch_losses))

            if self.verbose:
                print(
                    f'Epoch: {epoch}\n \
                        {time.time() - start_time} seconds\n \
                        {self.train_losses_[-1]}\n'
                )

        self.fit_time = time.time() - start_time

        return self


    def predict(self, X, return_logits=False):
        """
        Args:
            X: A pytorch tensor.
        """
        self.model.eval()

        with torch.no_grad():
            X = X.to(self.device)
            logits = self.model(X)

            _, predicted = torch.max(logits.data, 1)

            if return_logits:
                return predicted.cpu().numpy(), logits
            else:
                return predicted.cpu().numpy()


    def _backprop(self, X_batch, y_batch):
        self.optimizer.zero_grad()

        logits = self.model(X_batch)
        batch_loss = self.criterion(logits, y_batch)
        loss = batch_loss.item()

        batch_loss.backward()
        self.optimizer.step()

        return loss