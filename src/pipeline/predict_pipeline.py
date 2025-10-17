from .train_pipeline import train_main

def predict_main(model=None, feature_test=None, y_test=None, config=None):
    """Make a prediction."""
    if model is None:
        # 1-7 Implement the model
        model, _, feature_test_scaled, _, _ = train_main()

    # 8. Make a prediction
    predictions = model.predict(feature_test_scaled)
    
    return predictions


if __name__ == "__main__":
    predict_main()