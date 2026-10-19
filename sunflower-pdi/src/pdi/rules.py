import yaml
import json

def load_rules(config_path="configs/pdi_rules.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def pdi_proxy_label(cls, disc_ratio, area_ratio, sat_mean):
    """
    Rule-based PDI proxy.
    Returns: "Low", "Medium", "High"
    """
    if cls in ("YoungBud", "MatureBud"):
        return "High"
    if cls == "EarlyBloom":
        return "Medium" if disc_ratio < 0.30 else "Low"
    if cls == "Healthy":
        return "Low" if (disc_ratio > 0.12 and area_ratio > 0.15) else "Medium"
    if cls == "Wilted":
        return "Medium" if disc_ratio > 0.15 else "High"
    return "Medium"
