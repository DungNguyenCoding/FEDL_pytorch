import torch
import torch.nn as nn

class SVM(nn.Module):
    def __init__(self, input_dim=3072, num_classes=2):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 6 layers as specified in Source [480]
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.relu(self.fc1(x))
        return self.fc2(x)