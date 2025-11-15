import yaml
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class Config:
    raw: Dict[str, Any]

    def get(self, path: str, default=None):
        """
        Retrieve nested config using dotted path (e.g., 'paths.processed_dir').
        """
        cur = self.raw
        for p in path.split("."):
            if p in cur:
                cur = cur[p]
            else:
                return default
        return cur

def load_config(path: str = "configs/default.yaml") -> Config:
    """
    Load YAML and wrap in Config.
    """
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return Config(y)
