import os
import yaml

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")

def load_config(config_name):
    if not config_name.endswith(".yaml") and not config_name.endswith(".yml"):
        config_name += ".yml"
    config_path = os.path.join(CONFIGS_DIR, config_name)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
