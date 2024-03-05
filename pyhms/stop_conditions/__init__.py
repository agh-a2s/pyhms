from .gsc import (
    AllStopped,
    FitnessEvalLimitReached,
    GlobalStopCondition,
    NoActiveNonrootDemes,
    RootStopped,
    SingularProblemEvalLimitReached,
    SingularProblemPrecisionReached,
    WeightingStrategy,
)
from .lsc import AllChildrenStopped, FitnessSteadiness, LocalStopCondition
from .usc import DontRun, DontStop, MetaepochLimit, UniversalStopCondition
