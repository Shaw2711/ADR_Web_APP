import numpy as np
from data_loader import loader


def tanimoto(a, b):

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    intersection = np.sum(a * b)
    union = np.sum(a) + np.sum(b) - intersection

    if union == 0:
        return 0.0

    return intersection / union


def compute_similarity(fp, fp_train):
   
    sims = []

    for row in fp_train:
        sims.append(tanimoto(fp, row))

    return np.array(sims)


def filter_by_threshold(similarities, threshold):
   
    return np.where(similarities >= threshold)[0]