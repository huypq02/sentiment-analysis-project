import pandas as pd


class DataLoader:
    def load_csv(self, file):
        """
        Load CSV file into pandas DataFrame.
        
        :param file: Path to CSV file
        :type file: str
        :return: Loaded data
        :rtype: pandas.DataFrame
        """
        return pd.read_csv(file)
