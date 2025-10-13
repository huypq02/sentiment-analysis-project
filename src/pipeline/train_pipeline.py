import yaml
import os
from sklearn.model_selection import train_test_split
from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor

def load_config(path):
    """Load config safely with error handling."""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{path}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML config file '{path}': {e}")
        return None
    except Exception as e:
        print(f"Error: Unexpected error loading config file '{path}': {e}")
        return None

def main():
    """The training pipeline on the model"""
    # 1. Load config
    config = load_config('config/config.yaml')
    if config is None:
        print("Failed to load configuration. Exiting pipeline.")
        return
    filename = os.path.join(config['file']['directory'], config['file']['name'])

    # 2. Load data
    # Define a DataLoader's object
    loader = DataLoader()
    # Import dataset
    df = loader.load_csv(filename)
    
    # 3. Preprocessing
    # Preprocess data
    preprocessor = Preprocessor()
    df['reviewText_clean'] = df['reviewText'].apply(preprocessor.preprocess)
    # Convert df['reviewText_clean'] from tokens to string X
    texts_cleaned = df['reviewText_clean'].apply(
        lambda x: ' '.join(x)
    )
    labels = df['rating']

    # 4. Split
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(texts_cleaned, labels, 
                                                        test_size=0.2, random_state=0)

    # TODO: Will implement 5. Extractor Features
    

if __name__ == "__main__":
    main()