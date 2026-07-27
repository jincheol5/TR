import pandas as pd
import numpy as np
import torch
from .GNN_graph import GNN_Graph

class TGN_Graph(GNN_Graph):
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

    def get_temporal_neighbor(self,
            tar:torch.Tensor,
            tar_t:torch.Tensor,
            n_neighbor:int
        ):
        """
        Input:
            tar: [B,]
            tar_t: [B,]
            n_neighbor: int
            seed: int
        Return:
            neighbor: [B,num_neighbor]
            neighbor_t: [B,num_neighbor]
            neighbor_ts: [B,num_neighbor]
            neighbor_edge: [B,num_neighbor]
        """
        device=tar.device
        batch_size=tar.size(0)
        neighbor=torch.zeros(
            (batch_size,n_neighbor),
            dtype=torch.long,
            device=device
        )
        neighbor_t=torch.zeros(
            (batch_size,n_neighbor),
            dtype=torch.float,
            device=device
        )
        neighbor_ts=torch.zeros(
            (batch_size,n_neighbor),
            dtype=torch.float,
            device=device
        )
        neighbor_edge=torch.zeros(
            (batch_size,n_neighbor),
            dtype=torch.long,
            device=device
        )
        for b in range(batch_size):
            tar_id=int(tar[b].item())
            cut_time=float(tar_t[b].item())

            neighbors=self.adj.get(tar_id,[])
            times=self.adj_t.get(tar_id,[])

            if len(neighbors)==0:
                continue

            # t < cut_time 인 마지막 위치까지 선택
            times_np=np.asarray(times,dtype=np.float32)
            cut_idx=np.searchsorted(
                times_np,
                cut_time,
                side="left"
            )

            # 최근 num_neighbor개만 선택
            start_idx=max(0,cut_idx-n_neighbor)
            selected_neighbors=neighbors[start_idx:cut_idx]
            selected_times=times[start_idx:cut_idx]

            # 앞은 0 padding, 뒤에 실제 neighbor 저장
            offset=n_neighbor-len(selected_neighbors)
            for idx,((src,e_id),t) in enumerate(zip(selected_neighbors,selected_times)):
                neighbor[b,offset+idx]=src
                neighbor_t[b,offset+idx]=t
                neighbor_ts[b,offset+idx]=abs(cut_time-t)
                neighbor_edge[b,offset+idx]=e_id
        return {
            "neighbor":neighbor,
            "neighbor_t":neighbor_t,
            "neighbor_ts":neighbor_ts,
            "neighbor_edge":neighbor_edge
        }