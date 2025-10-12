import yaml
from sklearn.model_selection import train_test_split
from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor

def load_config(path):
    """Load config safely."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """The training pipeline on the model"""
    # 1. Load config
    config = load_config('config/config.yaml')
    filename = config['file']['path']+'/'+config['file']['name']

    # 2. Load data
    # Define a DataLoader's object
    loader = DataLoader()
    # Import dataset
    df = loader.load_csv(filename)
    
    # 3. Preprocessing
    # Preprocess data
    preprocessor = Preprocessor()
    df['reviewText_clean'] = df.apply(
        lambda x: preprocessor.preprocess(x['reviewText']),
        axis=1,
    )
    # Convert df['reviewText_clean'] from tokens to string X
    texts_cleaned = df['reviewText_clean'].apply(
        lambda x: ' '.join(x)
    )
    labels = df['rating']

    # 4. Split
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(texts_cleaned, labels, 
                                                        test_size=0.2, random_state=0)

    # 5. Extractor Features


if __name__ == "__main__":
    main()