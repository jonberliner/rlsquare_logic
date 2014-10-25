#RLSquare Logic!
##contains:
- action selection functions library
- an RL object class (only initializes so far)

##will contain:
- learning functions library
- training methods in RL objects
- RLSquare subclass

##example:
```
from rlsquare_logic.rlobjects import RLSquare
from numpy.random import rand
nSamples = 100

#  init RLSquare
rlsquare = RLSquare()

# generate some random state-reward pairs
randomStateParams = [rlsquare.sample() for _ in xrange(nSamples)]
randomStateRewards = [rand(nSamples) for _ in xrange(nSamples)]

# condition on state-reward pairs
rlsquare.condition(randomStateParams, randomStateRewards)

#  return params for your ultimate RLSquare!
conditionedRLSquareParams = rlsquare.sample()
```
