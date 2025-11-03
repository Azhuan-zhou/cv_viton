import yaml
from pathlib import Path

def load_config():
    path = Path(__file__).parent / "application.yaml"
    with open(path) as f:
        return yaml.safe_load(f)

CONFIG = load_config()
