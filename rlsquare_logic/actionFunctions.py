from numpy import exp, sum, ones
from numpy.random import RandomState
rng = RandomState()

def softmax(values, temp):
    N = len(values)
    raisedVals = exp(values / temp)
    probs = raisedVals / sum(raisedVals)
    action = rng.choice(N, p=probs)
    return action


def egreedy(values, epsilon):
    N = len(values)
    greedy = rng.rand() < epsilon
    best = values.argmax()
    if greedy:
        action = best
    else:
        probs = ones(N)
        probs[best] = 0
        probs /= probs.sum()
        action = rng.choice(N, p=probs)
    return action


def softmax_builder(temp):
    return lambda v: softmax(v, temp)


def egreedy_builder(e):
    return lambda v: egreedy(v, e)
