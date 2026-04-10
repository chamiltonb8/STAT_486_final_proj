import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

def load_csv(path):
    return pd.read_csv(os.path.join(BASE_DIR, path))