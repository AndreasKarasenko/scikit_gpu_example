from cuml.linear_model import LogisticRegression as cuLR


def model():
    """Return a linear model."""
    return cuLR(penalty="l2")