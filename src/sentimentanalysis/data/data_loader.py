import pandas as pd


class DataLoader:
    def load_csv(self, file):
        return pd.read_csv(file)
