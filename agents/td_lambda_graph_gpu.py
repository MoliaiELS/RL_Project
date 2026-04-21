from .graph_encoder import GraphEncoder as GraphEncoderGPU
from .td_lambda_graph import (
    ActionValueHead,
    GraphActionValueNetwork,
    TDLambdaGraphAgent,
)


class ActionValueHeadGPU(ActionValueHead):
    pass


class TDLambdaGraphAgentGPU(TDLambdaGraphAgent):
    pass


__all__ = [
    "ActionValueHeadGPU",
    "GraphActionValueNetwork",
    "GraphEncoderGPU",
    "TDLambdaGraphAgentGPU",
]
