from dataclasses import dataclass
from typing import Optional
from .constants import DEFAULT_CONFIG_PATH


@dataclass
class DataParameters:
    """
    Configuration for dataset input.

    Attributes:
        data_path (Optional[str]): Path to the dataset file.
        text_column (str): Name of the column containing text data. Default is 'reviewText'.
        label_column (str): Name of the column containing labels/targets. Default is 'rating'.
    """
    data_path: Optional[str] = None
    text_column: str = "reviewText"
    label_column: str = "rating"

@dataclass
class ComponentSelection:
    """
    Selection of feature extractor and model components.

    Attributes:
        extractor_name (str): Name of the feature extractor to use (e.g., 'tfidf', 'bow').
        model_name (str): Name of the model to use (e.g., 'logreg', 'naive_bayes').
    """
    extractor_name: str = "tfidf"
    model_name: str = "logreg"

@dataclass
class Hyperparameters:
    """
    Hyperparameters for model and feature extractor.

    Attributes:
        extractor_params (Optional[dict]): Parameters for the feature extractor.
        model_params (Optional[dict]): Parameters for the model.
    """
    extractor_params: Optional[dict] = None
    model_params: Optional[dict] = None
    param_grid: Optional[dict] = None

@dataclass
class TrainingConfiguration:
    """
    Training configuration settings.

    Attributes:
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Random seed for reproducibility. Default is 0.
        feature_scaling (bool): Whether to apply feature scaling. Default is False.
    """
    test_size: float = 0.2
    random_state: int = 0
    feature_scaling: bool = False

@dataclass
class FilePaths:
    """
    File path configuration.

    Attributes:
        config_path (str): Path to the main configuration YAML file. Default is 'config/config.yaml'.
        save_dir (Optional[str]): Directory to save outputs, models, or results.
    """
    config_path: str = DEFAULT_CONFIG_PATH
    save_dir: Optional[str] = None

@dataclass
class MLFlowTracking:
    """
    MLflow experiment tracking configuration.

    Attributes:
        mlflow_tracking (bool): Whether to enable MLflow tracking. Default is True.
        experiment_name (str): Name of the MLflow experiment. Default is 'sentiment-analysis'.
        run_name (Optional[str]): Name of the MLflow run.
        tags (Optional[str]): Tags for the MLflow run.
    """
    mlflow_tracking: bool = True
    experiment_name: str = "sentiment-analysis"
    run_name: Optional[str] = None
    tags: Optional[dict] = None
