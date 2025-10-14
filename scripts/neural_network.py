from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        # Input layer: 15-20 neurons (raycasts + speed + angle)
        # Hidden layers: 2-3 layers with 32-64 neurons each
        # Output layer: 4 neurons (one per control direction)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Sigmoid(),  # Added sigmoid for BCELoss
        )

    def forward(self, x):
        x = self.flatten(x)
        output = self.linear_relu_stack(x)
        return output