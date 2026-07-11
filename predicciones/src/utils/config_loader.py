import yaml
from pathlib import Path

def load_config():
    config_path = Path(__file__).parent.parent.parent / "configs" / "model_config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()
