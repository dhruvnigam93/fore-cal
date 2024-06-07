import torch
from torch import nn, optim
import numpy as np
from torch.nn.functional import cross_entropy
from sklearn.metrics import log_loss


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """

    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        return self


from scipy.optimize import minimize
from scipy.special import logit, expit


class TemperatureScaler:
    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, probs: np.ndarray, labels: np.ndarray):
        probs = np.array(probs) if isinstance(probs, list) else probs
        labels = np.array(labels) if isinstance(labels, list) else labels
        logits = logit(probs)

        def nll(temp: float) -> float:
            if temp <= 0:
                return np.inf
            scaled_logits = logits / temp
            scaled_probs = expit(scaled_logits)
            return log_loss(labels, scaled_probs)

        res = minimize(nll, [1.0], bounds=[(0.1, None)])
        self.temperature = res.x[0]

    def predict(self, probs: np.ndarray) -> np.ndarray:
        logits = logit(probs)
        scaled_logits = logits / self.temperature
        return expit(scaled_logits)


class HistogramBinning:
    def __init__(self, bins=5):
        self.bins = bins
        self.bin_boundaries = None
        self.bin_values = None

    def fit(self, predicted_probabilities, labels):
        # Determine the bin boundaries
        self.bin_boundaries = np.linspace(0, 1, self.bins + 1)
        bin_indices = np.digitize(np.clip(predicted_probabilities - 0.0001, 0, None), self.bin_boundaries) - 1
        bin_sums = np.zeros(self.bins)
        bin_counts = np.zeros(self.bins)

        for bin_index, label in zip(bin_indices, labels):
            bin_sums[bin_index] += label
            bin_counts[bin_index] += 1

        self.bin_values = bin_sums / np.maximum(bin_counts, 1)  # Avoid division by zero
        return self

    def predict(self, predicted_probabilities):
        bin_indices = np.digitize(np.clip(predicted_probabilities - 0.0001, 0, None), self.bin_boundaries) - 1
        return self.bin_values[bin_indices]


class BBQ:
    def __init__(self, bins_list=[5, 10, 50, 100]):
        self.bins_list = bins_list
        self.models = []
        self.weights = []

    def fit(self, predicted_probabilities, labels):
        from sklearn.metrics import log_loss

        self.models = [HistogramBinning(bins=b) for b in self.bins_list]
        log_losses = []

        for model in self.models:
            model.fit(predicted_probabilities, labels)
            calibrated_probs = model.predict(predicted_probabilities)
            log_losses.append(log_loss(labels, calibrated_probs))

        min_log_loss = min(log_losses)
        self.weights = [np.exp(-0.5 * (ll - min_log_loss)) for ll in log_losses]
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

        return self

    def predict(self, predicted_probabilities):
        calibrated_probs = np.zeros_like(predicted_probabilities)

        for weight, model in zip(self.weights, self.models):
            calibrated_probs += weight * model.predict(predicted_probabilities)

        return calibrated_probs
