import numpy as np
from numpy.random import RandomState
rng = RandomState()

def softmax(values, temp):
    N = len(values)
    raisedVals = np.exp(values / temp)
    probs = raisedVals / np.sum(raisedVals)
    action = rng.choice(N, p=probs)
    return action


def egreedy(values, epsilon):
    greedy = rng.rand() < epsilon
    best = values.argmax()
    N = len(values)
    if greedy:
        action = best
    else:
        probs = np.ones(N)
        probs[best] = 0
        probs /= probs.sum()
        action = rng.choice(N, p=probs)
    return action
