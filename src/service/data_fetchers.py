import pandas as pd
from src.domain.entities import TrainingDataset
from typing import Literal

Source = Literal["uri", "local", "s3"]


class DataFetcher:
    def __init__(self, path: str, source: Source):
        self.path = path

    def fetch_and_prepare_data(self) -> TrainingDataset:
        # Fetch the dataset
        df = pd.read_csv(self.path)

        # Assuming the last column is the label and the rest are features
        # This might need to be adjusted depending on the dataset structure
        features = df.columns.tolist()[0:-1]
        labels = df.columns.tolist()[-1]

        # Wrap in a TrainingData object
        training_data = TrainingDataset(data_id=self.path, features=features, labels=labels, data=df)
        return training_data

# # Example usage
# if __name__ == "__main__":
#     # Example URL to a dataset in CSV format from UCI ML Repository
#     # You should replace this URL with the actual dataset URL you're interested in
#     url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#     data_fetcher = DataFetcherService(url)
#     training_data = data_fetcher.fetch_and_prepare_data()
#     print(training_data)
