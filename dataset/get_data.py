import numpy as np
import pandas as pd


# Example feature configuration
# Users can modify these according to their own datasets
FEATURE_COLUMNS = [
    "feature_1",
    "feature_2",
    "feature_3",
]

TARGET_COLUMN = "target"


def get_data(dataframe_list, auxiliary_data=None):
    """
    Prepare input features and labels.

    Parameters
    ----------
    dataframe_list : list
        A list of pandas DataFrames containing input features.

    auxiliary_data : optional
        Additional environmental information.

    Returns
    -------
    feature_list : list
        Model input features.

    label_list : list
        Target values.
    """

    feature_list = []
    label_list = []


    for df in dataframe_list:

        # Extract input features
        features = df[FEATURE_COLUMNS].values


        # Extract target labels
        labels = df[TARGET_COLUMN].values


        feature_list.append(features)

        label_list.append(labels)


    return feature_list, label_list