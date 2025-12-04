import os
import joblib
from sklearn.model_selection import train_test_split
from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.features.tfidf_extractor import TFIDFExtractor
from src.models.logreg_model import LogisticRegressionModel
from src.utils.load_config import load_config
from src.utils.logger import setup_logging

logger = setup_logging(__name__)

def train_main(config_path='config/config.yaml'):
    """The training pipeline on the model"""
    # 1. Load config
    logger.info("Loading configuration...")
    config = load_config(config_path)
    if config is None:
        logger.error("Failed to load configuration. Exiting.")
        raise RuntimeError("Failed to load configuration.")
    filename = os.path.join(config['file']['raw_dir'], config['file']['name'])

    try:
        # 2. Load data
        logger.info("Loading data...")
        # Define a DataLoader's object
        loader = DataLoader()
        # Import dataset
        df = loader.load_csv(filename)
        
        # 3. Preprocessing
        logger.info("Data preprocessing...")
        # Preprocess data
        preprocessor = Preprocessor()
        df['reviewText_clean'] = df['reviewText'].apply(preprocessor.preprocess)
        # Convert df['reviewText_clean'] from tokens to string X
        texts_cleaned = df['reviewText_clean'].apply(
            lambda x: ' '.join(x)
        )
        labels = df['rating']

        # 4. Split
        logger.info("Splitting dataset...")
        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(texts_cleaned, labels, 
                                                            test_size=0.2, random_state=0)

        # 5. Extractor Features
        logger.info("Implementing the extractor feature...")
        # TODO: consider loading config of a specific feature
        extractor = TFIDFExtractor()
        feature_train = extractor.fit_transform(X_train)
        feature_test = extractor.transform(X_test)

        # 6. Model strategy
        logger.info("Implementing the model...")
        # TODO: consider loading config of a specific model
        model = LogisticRegressionModel()
        # TODO: Consider if-else with the model no need feature scaling
        feature_train_scaled, feature_test_scaled = model.scale_feature(feature_train, feature_test) # Feature scaling
        model.train(feature_train_scaled, y_train) # Train data on the model

    except Exception as e:
        logger.exception(f'Unexpected error in training pipeline: {e}')
        return None

    # 7. Save model and feature extractor
    # Create new folder with the name 'models' if it doesn't exist
    os.makedirs(config['model']['dir'], exist_ok=True)

    logger.info("Saving model and extractor...")
    # Dump files
    joblib.dump(model, config['models']['model'])
    joblib.dump(extractor, config['models']['extractor'])

    return model, extractor, feature_test_scaled, y_test, config

if __name__ == "__main__":
    train_main()