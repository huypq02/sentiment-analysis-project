import argparse
import os
import joblib
import logging
from sentimentanalysis import __version__, DEFAULT_CONFIG_PATH
from sentimentanalysis.pipeline import train, predict, evaluate_saved_model, SentimentPipeline
from sentimentanalysis.config import (
    DataParameters,
    ComponentSelection,
    Hyperparameters,
    TrainingConfiguration,
    FilePaths,
    MLFlowTracking,
)
from sentimentanalysis.utils import load_config, setup_logging


def _configure_command_logging(args, logger_names):
    """Configure log levels for command-related loggers."""
    if args.debug:
        level = logging.DEBUG
    elif args.verbose:
        level = logging.INFO
    else:
        return

    for logger_name in logger_names:
        setup_logging(name=logger_name, level=level)


def train_command(args):
    _configure_command_logging(args, [__name__, "sentimentanalysis.pipeline.training", "sentimentanalysis.pipeline.evaluation"])
    
    config_path = os.environ.get("CONFIG_PATH", args.config)
    config = load_config(config_path)
    
    train(
        data_params=DataParameters(
            data_path=args.data_path or config["data"]["data_path"]
        ),
        component_sel=ComponentSelection(
            extractor_name=args.extractor or config["components"]["extractor_name"],
            model_name=args.model or config["components"]["model_name"]
        ),
        hyperparams=Hyperparameters(
            extractor_params=config["hyperparameters"].get("extractor_params"),
            model_params=config["hyperparameters"].get("model_params"),
            param_grid=config["hyperparameters"].get("param_grid")
        ),
        training_conf=TrainingConfiguration(
            test_size=args.test_size or config["training"]["test_size"],
            random_state=args.random_state or config["training"]["random_state"],
            feature_scaling=args.feature_scaling or config["training"]["feature_scaling"],
            evaluate_after_training=not args.no_eval
        ),
        file_paths=FilePaths(config_path=args.config),
        mlflow_tracking=MLFlowTracking()
    )

def predict_command(args):
    _configure_command_logging(args, [__name__, "sentimentanalysis.pipeline.prediction"])

    config_path = os.environ.get("CONFIG_PATH", args.config)
    config = load_config(config_path)
    model_path = config["models"]["model"]
    extractor_path = config["models"]["extractor"]

    if os.path.exists(model_path) and os.path.exists(extractor_path):
        model = joblib.load(config["models"]["model"])
        extractor = joblib.load(config["models"]["extractor"])

    pipeline = SentimentPipeline(extractor, model)

    predict(
        pipeline=pipeline,
        feature=args.text
    )

def evaluate_command(args):
    _configure_command_logging(args, [__name__, "sentimentanalysis.pipeline.evaluation"])
    
    evaluate_saved_model(
        config_path=args.config, 
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state,
        text_column=args.text_column,
        label_column=args.label_column
    )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="sentimentanalysis",
        description="Sentiment Analysis CLI - Train, evaluate, and predict sentiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with defaults
  %(prog)s train
  
  # Train with custom settings
  %(prog)s train --model logreg --extractor tfidf --feature-scaling
  
  # Evaluate saved model
  %(prog)s evaluate
  
  # Make a prediction
  %(prog)s predict "This movie is amazing!"
  
  # Show version
  %(prog)s --version

For more information, visit: https://github.com/huypq02/sentiment-analysis-project
"""
    )

    parser.add_argument("-v", "--version", 
                        action="version", 
                        version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", 
                                       description="valid subcommands",
                                       help="additional help")
    # Train commands
    train_parser = subparsers.add_parser(
        "train", 
        help="Train model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    train_parser.add_argument("--config", 
                              help="configuration path", 
                              default=DEFAULT_CONFIG_PATH)
    train_parser.add_argument("--data-path", type=str, help="Dataset path")
    train_parser.add_argument("--model",
                              type=str,
                              choices=("logreg", "naive_bayes"), 
                              help="Model types", default="logreg")
    train_parser.add_argument("--extractor",
                              type=str,
                              choices=("tfidf", "bow"),
                              help="Feature extractor types", default="tfidf")
    train_parser.add_argument("--test-size",
                              type=float,
                              help="Test size for training model",
                              default=0.2)
    train_parser.add_argument("--random-state",
                              type=int,
                              help="Random state for training model",
                              default=0)
    train_parser.add_argument("--feature-scaling",
                              action='store_true',
                              help="Enable feature scaling if applicable",
                              default=False)
    train_parser.add_argument("--no-eval",
                              action='store_true',
                              help="Skip evaluation after training",
                              default=False)
    train_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    train_parser.add_argument("--debug", action="store_true", help="Debug mode")
    train_parser.add_argument("--text-column", type=str, help="Name of text column in dataset", default=None)
    train_parser.add_argument("--label-column", type=str, help="Name of label column in dataset", default=None)
    train_parser.set_defaults(func=train_command)

    # Predict commands
    predict_parser = subparsers.add_parser(
        "predict", 
        help="Predict target",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    predict_parser.add_argument("--config", 
                              help="configuration path", 
                              default=DEFAULT_CONFIG_PATH)
    predict_parser.add_argument("text", type=str, help="Review text")
    predict_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    predict_parser.add_argument("--debug", action="store_true", help="Debug mode")
    predict_parser.set_defaults(func=predict_command)

    # Evaluate commands
    evaluate_parser = subparsers.add_parser(
        "evaluate", 
        help="Evaluate model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    evaluate_parser.add_argument("--config", 
                                 help="configuration path", 
                                 default=DEFAULT_CONFIG_PATH)
    evaluate_parser.add_argument("--data-path", type=str, help="Dataset path")
    evaluate_parser.add_argument("--test-size",
                                 type=float,
                                 help="Test size for training model",
                                 default=0.2)
    evaluate_parser.add_argument("--random-state",
                                 type=int,
                                 help="Random state for training model",
                                 default=0)
    evaluate_parser.add_argument("--text-column", type=str, help="Name of text column in dataset", default=None)
    evaluate_parser.add_argument("--label-column", type=str, help="Name of label column in dataset", default=None)
    evaluate_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    evaluate_parser.add_argument("--debug", action="store_true", help="Debug mode")
    evaluate_parser.set_defaults(func=evaluate_command)
    
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
    else:
        args.func(args)

if __name__ == "__main__":
    main()
