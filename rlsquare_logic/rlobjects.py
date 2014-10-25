from numpy import isscalar, isvector, ones_like, linspace
from actionFunctions import softmax_builder
from learningFunctions import td0_builder



# class GPObject(object):
#     def __init__(self, aquisitionFunction, priorMu, priorCovmat):
#         self.stateSpace =
#         if not priorMu:


class RLObject(object):
    def __init__(self, stateSpace, actionFunction, learningFunction, initValue=0.):
        for k, v in stateSpace.iteritems():
            assert isvector(v), "all fields in stateSpace must be vectors"
        self._stateSpace = stateSpace
        self.value = initValue
        self.actionFunction = actionFunction
        self.learningFunction = learningFunction

    def sample(self):
        ichoice = {k: self.actionFunction(v) for k, v in self.values.iteritems()}
        params = {k: self.stateSpace[k][i] for k, i in ichoice.iteritems()}
        return params

    def condition(self, action, reward):
        for k, v in self.value.iteritems():
            self.value[k] = self.learningFunction(v, action, reward)

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
            stateSpace[cc] = linspace(0, 255, 17)[1:]
        stateSpace['oscTime'] = linspace(0, 16000, 17)[1:]
        stateSpace['sigSteepness'] = linspace(0, 4, 17)[1:]
        RLObject.__init__(self, stateSpace, actionFunction, learningFunction,
                          initValue)

