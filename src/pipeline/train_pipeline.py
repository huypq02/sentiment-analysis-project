import yaml
import os
import joblib
from sklearn.model_selection import train_test_split
from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.features.tfidf_extractor import TFIDFExtractor
from src.models.logreg_model import LogisticRegressionModel

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
        exit(1)
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

    # 5. Extractor Features
    # TODO: consider loading config of a specific feature
    extractor = TFIDFExtractor()
    feature_train = extractor.fit_transform(X_train)
    feature_test = extractor.transform(X_test)

    # 6. Model strategy
    # TODO: consider loading config of a specific model
    model = LogisticRegressionModel()
    # TODO: Consider if-else with the model no need feature scaling
    feature_train_scaled, feature_test_scaled = model.scale_feature(feature_train, feature_test) # Feature scaling
    model.train(feature_train_scaled, y_train) # Train data on the model

    # 7. Evaluation
    cm, accuracy = model.evaluate(feature_test_scaled, y_test)
    print("Confusion Matrix:\n", cm)
    print("Accuracy Score:", accuracy)

    # 8. Save model and feature extractor
    joblib.dump(model, "models/sentiment_logreg.pkl")
    joblib.dump(extractor, "models/tfidf_extractor.pkl")

    # 9. (Optional) Log results
    with open("models/metrics.txt", "w") as f:
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nAccuracy Score: ")
        f.write(str(accuracy))

if __name__ == "__main__":
    main()