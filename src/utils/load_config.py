import yaml

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