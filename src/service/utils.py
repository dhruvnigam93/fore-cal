import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import BinaryEncoder
from typing import List, Tuple
from typeguard import typechecked


@typechecked
def convert_to_binary_dataset(data_features: pd.DataFrame, data_label: pd.Series) -> Tuple[
    pd.DataFrame, List[str], pd.Series]:
    """
    Convert a dataset into a format suitable for binary classification using binary encoding for categorical features.

    Parameters:
    - data_features: A pandas DataFrame containing the dataset.
    - data_label: A pandas Series containing the label.

    Returns:
    - A tuple containing:
        - The modified DataFrame ready for binary classification.
        - The updated list of feature names after binary encoding.
        - The label column name.
    """
    data_features = data_features.copy()
    data_features = data_features.loc[:, ~data_features.columns.duplicated()]
    data_label = data_label.copy()
    original_label_name = data_label.name
    unique_labels = data_label.unique()

    if len(unique_labels) > 2:
        most_common_label = data_label.value_counts().idxmax()
        print(f"Number of unique labels - {len(unique_labels)}. Most common label: {most_common_label}")
        data_label = data_label.apply(lambda x: 1 if x == most_common_label else 0)

    if (data_label.dtype == 'object') or (data_label.dtype.name == 'category') or (set(unique_labels) != {0, 1}):
        le = LabelEncoder()
        data_label_encoded = le.fit_transform(data_label)
        if len(le.classes_) != 2:
            raise ValueError("Expected binary labels after encoding.")
        data_label = pd.Series(data_label_encoded, name=original_label_name)

    categorical_features = [feature for feature in data_features.columns if
                            feature in data_features.select_dtypes(
                                include=['object', 'category']).columns.tolist()]

    non_categorical_features = [col for col in data_features.columns if col not in categorical_features]
    data_features[non_categorical_features] = data_features[non_categorical_features].fillna(
        data_features[non_categorical_features].mean())

    for column in categorical_features:
        data_features[column] = data_features[column].fillna(data_features[column].mode().iloc[0])

    if categorical_features:
        encoder = BinaryEncoder(cols=categorical_features)
        data_features = encoder.fit_transform(data_features)

    new_features = list(data_features.columns)

    return data_features, new_features, data_label


@typechecked
def test_dataset_binary_numeric(data_features: pd.DataFrame, data_label: pd.Series):
    """
    Test if the dataset is suitable for binary classification task
    :param df:
    :param target_col:
    :return:
    """
    assert data_features.select_dtypes(include=[object]).empty, "Dataset contains non-numeric columns"
    assert data_label.nunique() == 2, "Target column is not binary"
    print("All tests passed!")
