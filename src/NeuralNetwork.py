import torch.nn as nn

class NeuralNetwork(nn.Module):
    """
    A simple neural network with n_layers hidden layers and an output layer.
    """
    def __init__(self, in_features, out_features, n_classes) -> None:
        super().__init__()
        self.module_list = [
          nn.Linear(
            in_features,
            out_features
          ),
          nn.ReLU(),
          nn.Linear(out_features, out_features),
          nn.ReLU(),
          nn.Linear(out_features, n_classes)
        ]

        self.network = nn.Sequential(*self.module_list)


    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): input tensor
        """
        return self.network(x)