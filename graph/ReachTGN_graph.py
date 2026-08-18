import pandas as pd
import numpy as np
import torch
from .TGN_graph import TGN_Graph

class ReachTGN_Graph(TGN_Graph):
    def __init__(self,
            graph_df:pd.DataFrame,
            node_ft:np.ndarray=None,
            edge_ft:np.ndarray=None,
            node_dim:int=32,
            edge_dim:int=32,
            bipartite:bool=False
        ):
        super().__init__(
            graph_df=graph_df,
            node_ft=node_ft,
            edge_ft=edge_ft,
            node_dim=node_dim,
            edge_dim=edge_dim,
            bipartite=bipartite
        )

    def Temporal_Augmentation(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor,
            edge:torch.Tensor,
            drop_rate:float=0.1,
            jitter_std:float=0.01,
            jitter_range:float=0.01,
        ):
        """
        ReaCH-TGN Temporal Augmentation

        1. Event Drop
        2. Timestamp Jitter
            - Gaussian noise
            - Uniform jitter
        
        Input:

        Return:
        """
        # Event Drop
        keep_mask=torch.rand(
            len(src),
            device=src.device
        )>=drop_rate

        src=src[keep_mask]
        dst=dst[keep_mask]
        event_t=event_t[keep_mask]
        edge=edge[keep_mask]

        # Gaussian timestamp jitter
        event_t=event_t+torch.randn_like(t)*jitter_std

        # Uniform timestamp jitter
        event_t=event_t+(torch.rand_like(t)*2-1)*jitter_range

        # timestamp 변화로 시간 순서가 바뀔 수 있으므로 재정렬
        order_idx=torch.argsort(event_t)

        return {
            "src":src[order_idx],
            "dst":dst[order_idx],
            "event_t":event_t[order_idx],
            "edge":edge[order_idx]
        }