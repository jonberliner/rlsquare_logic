import numpy as np

NCHANNELOPTIONS = 10

RGB1 = np.tile(np.linspace(0, 1, NCHANNELOPTIONS), [3, 1])
RGB2 = np.tile(np.linspace(0., 1., NCHANNELOPTIONS), [3, 1])

class RLObject(object):
    def __init__(self, stateSpace, actionFcn, learningRule, initValue=0.):
        self._stateSpace = stateSpace
        self.value = initValue
        self.actionFcn = actionFcn
        self.learningRule = learningRule

    # stateSpace has no setter because cannot change after initialization
    @property
    def stateSpace(self):
        return self._stateSpace

    @property
    def actionFcn(self):
        return self._actionFcn

    @actionFcn.setter
    def actionFcn(self, fcn):
        assert callable(fcn), "actionFcn must be a function"
        self._actionFcn = fcn

    @property
    def learningFcn(self):
        return self._learningFcn

    @learningFcn.setter
    def learningFcn(self, fcn):
        assert callable(fcn), "learningFcn must be a function"
        self._learningFcn = fcn

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        # initialize if haven't yet
        if not hasattr(self, '_value'): self._value = {}
        # scalar, set all values of all states of all stateSpace fields to value
        if np.isscalar(value):
            for k in self.stateSpace:
                self._value[k] = np.ones_like(self.stateSpace[k]) * value
        else:
            # you can set specfic fields of the statespace
            assert set(value.keys()).issubset(set(self.stateSpace.keys())),\
                    "value keys being set must be in stateSpace keys"
            for k, v in value.iteritems():
                # set value of all states of stateSpace field k to v
                if np.isscalar(v):
                    self._value[k] = np.ones_like(self.stateSpace[k]) * v
                # set values of states in stateSpace field k to those in v
                else:
                    assert v.shape == self.stateSpace[k].shape,\
                        ("if value['key'] is not a scalar, "
                         "must be same size as self.stateSpace[key]")
                    self._value[k] = v
