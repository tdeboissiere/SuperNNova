import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, input_size, settings):
        super(CNN, self).__init__()

        self.output_size = settings.nb_classes
        self.hidden_size = settings.hidden_dim
        last_input_size = settings.hidden_dim
        self.kernel_size = 3

        self.conv1 = nn.Conv1d(input_size, self.hidden_size, kernel_size=self.kernel_size,
                               stride=1, padding=(self.kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(self.hidden_size, last_input_size,
                               kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2)

        self.output_layer = torch.nn.Linear(last_input_size, self.output_size)

    def forward(self, x, mean_field_inference=False):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # mean pooling
        x = x.mean(2)

        output = self.output_layer(x)

        return output
