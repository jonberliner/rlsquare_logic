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
rlsl = RLSquare()  #  init RLSquare
rlsl.condition(observations)  #  condition on observations
rlSquareParams = rlsl.sample()  #  return params for your ultimate RLSquare!
```
