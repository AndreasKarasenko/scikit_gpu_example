### A function to load the data from data/samples/train.csv and data/samples/test.csv
import argparse
import json
from typing import List

import pandas as pd


from utils.preprocess import TextPreprocessor


def get_config_names(
    args: argparse.Namespace,
) -> tuple[List[str], List[str], List[str]]:
    """Get the dataset, input and target variable names from the config files."""
    datasets = json.load(open(args.data_config + "dataset_config.json", "r"))
    datasets = datasets["datasets"]

    input_vars = json.load(open(args.data_config + "input_config.json", "r"))
    input_vars = input_vars["input_var"]

    target_vars = json.load(open(args.data_config + "target_config.json", "r"))
    target_vars = target_vars["target_vars"]

    return datasets, input_vars, target_vars


def load_samples(
    dataset_name: str, input_var: str, target_var: str, args: argparse.Namespace
):
    """Loads samples generated by utils/create_samples.py
    By default samples share the same layout and require no further handling.
    Data loaded this way must still be preprocessed."""
    train = pd.read_csv(args.data_dir + dataset_name + "_train.csv")
    train.dropna(inplace=True)
    test = pd.read_csv(args.data_dir + dataset_name + "_test.csv")
    test.dropna(inplace=True)

    X_train = train[input_var]
    X_test = test[input_var]
    preprocessor = TextPreprocessor()
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    y_train = train[target_var]
    y_test = test[target_var]

    return X_train, X_test, y_train, y_test
