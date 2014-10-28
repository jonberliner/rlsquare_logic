#RLSquare Logic!
##contains:
- action selection functions library
- learning rule functions library
- an RL object class that:
    - trains on state-reward pairs
    - generates value-weighted random samples
- an RLSquare object subclass that:
    - has fixed RLObject statespace
    - has presets for learningFunction, actionFunction, values


##example:
```
from rlsquare_logic.rlobjects import RLSquare
from numpy.random import rand
nSamples = 100

#  init RLSquare
rlsquare = RLSquare()

# generate some random state-reward pairs
randomStateParams = [rlsquare.sample() for _ in xrange(nSamples)]
# randomStateRewards = rand(nSamples).tolist()
randomStateRewards = rand(nSamples)

# condition on state-reward pairs
rlsquare.condition(randomStateParams, randomStateRewards)

#  return params for your ultimate RLSquare!
conditionedRLSquareParams = rlsquare.sample()
```
