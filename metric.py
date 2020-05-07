import numpy as np
from scipy.spatial.distance import cdist
import utils
def pair_sim(X,Y):
    return 1 - cdist(X,Y,'cosine')

def get_max_pair(X,Y):
    X = X.value
    Y = np.array([y.value for y in Y])

    cos_sim = lambda x: np.dot(X, x) / (np.linalg.norm(X) * np.linalg.norm(x))
    return max(list(map(cos_sim, Y)))

def get_mean_sim(X,Y):
    X = X.value
    Y = np.array([y.value for y in Y])

    cos_sim = lambda x: np.dot(X, x) / (np.linalg.norm(X) * np.linalg.norm(x))
    return sum(list(map(cos_sim, Y)))/len(Y)



def get_min_sim(X,Y):
    X = X.value
    Y = np.array([y.value for y in Y])

    cos_sim = lambda x: np.dot(X, x) / (np.linalg.norm(X) * np.linalg.norm(x))
    return min(list(map(cos_sim, Y)))/len(Y)

def get_all_sim(X,Y):
    X = X.value
    Y = np.array([y.value for y in Y])

    cos_sim = lambda x: np.dot(X, x) / (np.linalg.norm(X) * np.linalg.norm(x))
    return list(map(cos_sim, Y))