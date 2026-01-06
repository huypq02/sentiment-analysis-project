import pandas as pd


class DataLoader:
    def __init__(self):
        pass

    def load_csv(self, file):
        return pd.read_csv(file)
