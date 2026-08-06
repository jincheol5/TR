import pandas as pd
import numpy as np
import torch
from .temporal_graph import TemporalGraph

class DyGFormer_Graph(TemporalGraph):
    def __init__(self,
            graph_df:pd.DataFrame,
            bipartite:bool=False
        ):
        super().__init__(
            graph_df=graph_df,
            bipartite=bipartite
        )