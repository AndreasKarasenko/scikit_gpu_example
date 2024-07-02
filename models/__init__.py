from models import (
    gpu_lr,
    gpu_nb,
    gpu_rf,
    naive_bayes,
    rf,
    logistic_regression,
)

MODELS = {
    # "MultinomialNB": naive_bayes.model,
    "LR": logistic_regression.model,
    # "RF": rf.model,
}

GPU_MODELS = {
    # "MultinomialNB": gpu_nb.model,
    # "LR": gpu_lr.model,
    "RF": gpu_rf.model,
}