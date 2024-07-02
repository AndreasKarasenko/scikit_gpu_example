from cuml.ensemble import RandomForestClassifier as cuRF


def model():
    """Return a random forest model."""
    return cuRF()