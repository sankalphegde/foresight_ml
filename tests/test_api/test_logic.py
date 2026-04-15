def calculate_risk_level(prob: float) -> str:
    """Replicated standalone logic for unit testing risk assignment."""
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"


def test_risk_level_logic():
    """Verify that probability scores are binned into the correct risk categories."""
    assert calculate_risk_level(0.95) == "High"
    assert calculate_risk_level(0.70) == "High"
    assert calculate_risk_level(0.55) == "Medium"
    assert calculate_risk_level(0.40) == "Medium"
    assert calculate_risk_level(0.25) == "Low"
