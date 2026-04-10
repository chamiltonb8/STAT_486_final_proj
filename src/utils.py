import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
DATA_DIR = os.path.join(BASE_DIR, "data")
SRC_DIR = os.path.join(BASE_DIR, "src")
MODELS_DIR = os.path.join(SRC_DIR, "models")


def rel_path(path):
    return os.path.join(BASE_DIR, path)


def data_path(filename):
    return os.path.join(DATA_DIR, filename)


def model_path(filename):
    return os.path.join(MODELS_DIR, filename)


def load_csv(path):
    return pd.read_csv(rel_path(path))