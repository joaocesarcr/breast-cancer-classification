import torch.nn as nn

# Generator
class Generator(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
          super(Generator, self).__init__()
          layers = []
          prev_size = input_size
          for size in hidden_sizes:
              layers.append(nn.Linear(prev_size, size))
              layers.append(nn.LeakyReLU(0.2))
              if not (size == hidden_sizes[-1]):
                layers.append(nn.Dropout(0.5))
              prev_size = size
          layers.append(nn.Linear(prev_size, output_size))
          layers.append(nn.Tanh())
          self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(Discriminator, self).__init__()
        layers = []
        prev_size = input_size
        output_size = 1
        for size in hidden_sizes:
          layers.append(nn.Linear(prev_size, size))
          layers.append(nn.LeakyReLU(0.2))
          layers.append(nn.Dropout(0.5))
          prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
      return self.layers(x)
