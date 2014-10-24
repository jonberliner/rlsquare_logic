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
from rlsquare_logic.actionFcns import egreedy
from rlsquare_logic.learningFcn import qLearning
rlsl = RLSquare(stateSpace, egreedy, qLearning, initVal=0.)
rlsl.condition(db_observations)
rlSquareParams = rlsl.sample()
```
