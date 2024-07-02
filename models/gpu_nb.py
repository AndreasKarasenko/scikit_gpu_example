from cuml.naive_bayes import MultinomialNB as cuNB

def model():
    """Return a naive bayes model."""
    return cuNB()