from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Tuple
import pandas as pd
import torch
from torch import nn

from src.service.utils import convert_to_binary_dataset


# Value Objects
@dataclass(frozen=True)
class ModelParameters:
    parameters: dict


@dataclass(frozen=True)
class MetricValue:
    name: str
    value: float


class TrainingDataset:
    def __init__(self, data_id: str, data: pd.DataFrame, features: List[str], labels: str):
        self.data_id = data_id
        self.data, self.features, self.labels = convert_to_binary_dataset(data, features, labels)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    def __str__(self):
        return f"TrainingDataset(data_id={self.data_id}, data size={self.data.shape}, features={self.features}, labels={self.labels})"


class CalibrationDataset:
    def __init__(self, data_id: str, data: pd.DataFrame, exogenous_features: List[str], base_probability: str,
                 label: str):
        self.data_id = data_id
        self.data = data,
        self.exogenous_features = exogenous_features
        self.base_probability = base_probability
        self.label = label

    @property
    def shape(self) -> Tuple[int, int]:
        # Implement shape property
        return self.data.shape


class BaseClassificationAlgorithm(ABC):
    def __init__(self, name: str, parameters: ModelParameters):
        self.name = name
        self.parameters = parameters

    @abstractmethod
    def apply(self, data: TrainingDataset) -> Tuple[pd.DataFrame, dict]:
        """
        Train the model and return the results
        :param data:
        :return: Tuple of the output data and the metrics
        """
        pass


class CalibrationAlgorithm(ABC):
    def __init__(self, name: str, parameters: ModelParameters):
        self.name = name
        self.parameters = parameters

    @abstractmethod
    def apply(self, dataset: CalibrationDataset) -> Tuple[pd.DataFrame, List[MetricValue]]:
        # Implement calibration logic
        pass


class TrainingPipeline:
    def __init__(self, training_data: TrainingDataset, base_algorithm: BaseClassificationAlgorithm,
                 calibration_algorithm: CalibrationAlgorithm):
        self.training_data = training_data
        self.base_algorithm = base_algorithm
        self.calibration_algorithm = calibration_algorithm

    def run(self) -> Tuple[List[MetricValue], List[MetricValue]]:
        # Implement training pipeline
        base_output_data, base_metrics = self.base_algorithm.apply(self.training_data)
        calibrated_output_data, calibrated_metrics = self.calibration_algorithm.apply(base_output_data)
        return base_metrics, calibrated_metrics


class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, output_size=1):
        super(BinaryClassifier, self).__init__()

        layers = []
        for i in range(hidden_layers):

            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))

            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        x = self.layers[-1](x)
        return torch.sigmoid(x)
