def td0(value, reward, stepsize):
    value += stepsize * (reward-value)
    return value


def td0_builder(stepsize):
    return lambda v, r: td0(v, r, stepsize)
