import pytest
from src.pdi.rules import pdi_proxy_label

def test_pdi_proxy_label_bud():
    assert pdi_proxy_label("YoungBud", 0.1, 0.1, 0.0) == "High"
    assert pdi_proxy_label("MatureBud", 0.5, 0.5, 0.0) == "High"

def test_pdi_proxy_label_early_bloom():
    assert pdi_proxy_label("EarlyBloom", 0.2, 0.2, 0.0) == "Medium"
    assert pdi_proxy_label("EarlyBloom", 0.4, 0.2, 0.0) == "Low"

def test_pdi_proxy_label_healthy():
    assert pdi_proxy_label("Healthy", 0.15, 0.2, 0.0) == "Low"
    assert pdi_proxy_label("Healthy", 0.10, 0.2, 0.0) == "Medium"
    assert pdi_proxy_label("Healthy", 0.15, 0.1, 0.0) == "Medium"

def test_pdi_proxy_label_wilted():
    assert pdi_proxy_label("Wilted", 0.2, 0.2, 0.0) == "Medium"
    assert pdi_proxy_label("Wilted", 0.1, 0.2, 0.0) == "High"
