import torch
from torch import autograd, nn
import torch.nn.functional as F


class LeNet_5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 5, padding=2)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(32, 48, 5, padding=2)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(914 * 48, 100)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 100)
        self.dropout3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class GradientReversalFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.lam * grad_output.neg(), None


rev_grad = GradientReversalFunction.apply


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 5, padding=2)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(32, 48, 5, padding=2)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.dropout1 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        return x


class LabelPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(48 * 914, 100)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 100)
        self.dropout3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = self.fc3(x)
        return x


class DomainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc4 = nn.Linear(48 * 914, 100)
        self.dropout4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(100, 1)

    def forward(self, x, lam):
        x = rev_grad(x, lam)
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        x = self.fc5(x)
        return x


class DANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.label_predictor = LabelPredictor()
        self.domain_classifier = DomainClassifier()

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.label_predictor(x)
