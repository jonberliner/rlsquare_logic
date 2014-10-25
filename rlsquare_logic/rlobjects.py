from numpy import isscalar, ones_like, linspace, where
from actionFunctions import softmax_builder
from learningFunctions import td0_builder
import pdb

# TODO: finish gaussian process rl object
# class GPObject(object):
#     def __init__(self, aquisitionFunction, priorMu, priorCovmat):
#         self.stateSpace =
#         if not priorMu:

class RLObject(object):
    # TODO: add documentation
    def __init__(self, stateSpace, actionFunction, learningFunction, initValue=0.):
        for k, v in stateSpace.iteritems():
            assert v.ndim == 1, "all fields in stateSpace must be vectors"
        self._stateSpace = stateSpace
        self.value = initValue
        self.actionFunction = actionFunction
        self.learningFunction = learningFunction

    def sample(self):
        ichoice = {k: self.actionFunction(v) for k, v in self.value.iteritems()}
        params = {k: self.stateSpace[k][i] for k, i in ichoice.iteritems()}
        return params

    # FIXME: vectorize condition so doesn't need to call _conditionOnOne a lot
    def _conditionOnOne(self, state, reward):
        assert set(state.keys()) == set(self.stateSpace.keys()),\
            "state keys must match RLObject.stateSpace keys"
        for field, v in state.iteritems():
            try:
                i = where(self.stateSpace[field]==v)[0][0]  # get index of state
            except:
                raise ValueError("state instance must be in RLObject.stateSpace")

            self._value[field][i] =\
                self.learningFunction(self.value[field][i], reward)

    def condition(self, states, rewards):
        assert type(states) is list,\
            "states must be a list of dicts with keys matching self.stateSpace"
        assert type(rewards) is list, "rewards must be a list of scalars"
        [self._conditionOnOne(states[i], rewards[i])
            for i in xrange(len(states))]

    # stateSpace has no setter because cannot change after initialization
    @property
    def stateSpace(self):
        return self._stateSpace

    @property
    def actionFunction(self):
        return self._actionFunction

    @actionFunction.setter
    def actionFunction(self, fcn):
        assert callable(fcn), "actionFunction must be a function"
        self._actionFunction = fcn

    @property
    def learningFunction(self):
        return self._learningFunction

    @learningFunction.setter
    def learningFunction(self, fcn):
        assert callable(fcn), "learningFunction must be a function"
        self._learningFunction = fcn

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        # initialize if haven't yet
        if not hasattr(self, '_value'): self._value = {}
        # scalar, set all values of all states of all stateSpace fields to value
        if isscalar(value):
            for k in self.stateSpace:
                self._value[k] = ones_like(self.stateSpace[k]) * value
        else:
            # you can set specfic fields of the statespace
            assert set(value.keys()).issubset(set(self.stateSpace.keys())),\
                    "value keys being set must be in stateSpace keys"
            for k, v in value.iteritems():
                # set value of all states of stateSpace field k to v
                if isscalar(v):
                    self._value[k] = ones_like(self.stateSpace[k]) * v
                # set values of states in stateSpace field k to those in v
                else:
                    assert v.shape == self.stateSpace[k].shape,\
                        ("if value['key'] is not a scalar, "
                         "must be same size as self.stateSpace[key]")
                    self._value[k] = v


class RLSquare(RLObject):
    def __init__(self,
                 actionFunction=softmax_builder(1.),
                 learningFunction=td0_builder(0.1),
                 initValue=0.):
        # define rlsquare statespace
        colorChannels = {'h1', 's1', 'v1', 'h2', 's2', 'v2'}
        stateSpace = {}
        for cc in colorChannels:
            stateSpace[cc] = linspace(0, 255, 16)
        stateSpace['oscTime'] = linspace(0, 8000, 17)[1:]
        stateSpace['sigSteepness'] = linspace(0, 4, 17)[1:]
        RLObject.__init__(self, stateSpace, actionFunction, learningFunction,
                          initValue)

