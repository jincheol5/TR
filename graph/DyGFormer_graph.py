import pandas as pd
import numpy as np
import torch
from .TGN_graph import TGN_Graph

class DyGFormer_Graph(TGN_Graph):
    """
    << STEP >>
    1. src와 dst에 대한 S_src, S_dst 계산
    2. S_node로부터 X_n, X_e, X_t 계산
    3. S_src와 S_dst로부터 C_src, C_dst 구한 후 X_src_c, X_dst_c 계산
    4. node의 X_n, X_e, X_t, X_c를 각각 patching 후 M_n, M_e, M_t, M_c 계산
    """
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

    def get_history_seq(self,
            node:torch.Tensor,
            event_t:torch.Tensor,
            n_neighbor:int
        ):
        """
        batch 내 각 노드의 1-hop history sequence를 구해 list로 반환.
        각 sequence에 자기 자신의 정보를 맨 앞에 추가한다.
        이때, edge_id는 0, timespan은 0.0으로 추가한다.

        Input:
            node: [B,]
            event_t: [B,]
            n_neighbor: int
        Return:
            node_seq_list: list of each node's history neighbor sequence 
            edge_seq_list: list of each node's history edge sequence
            ts_seq_list: list of each node's history timespan sequence
        """
        if n_neighbor<0:
            raise ValueError(
                f"n_neighbor must be non-negative: {n_neighbor}"
            )

        node_seq_list=[]
        edge_seq_list=[]
        ts_seq_list=[]
        for node_id,cut_time in zip(
                node.detach().cpu().numpy(),
                event_t.detach().cpu().numpy()
            ):
            node_id=int(node_id)
            cut_time=float(cut_time)
            neighbors=self.adj[node_id]
            edges=self.adj_edge[node_id]
            times=self.adj_t[node_id]

            cut_idx=int(np.searchsorted(
                times,
                cut_time,
                side="left"
            ))
            start_idx=max(0,cut_idx-n_neighbor)
            selected_neighbors=neighbors[start_idx:cut_idx]
            selected_edges=edges[start_idx:cut_idx]
            selected_times=times[start_idx:cut_idx]
            node_seq_list.append(
                np.concatenate([
                    np.asarray([node_id],dtype=np.int64),
                    selected_neighbors
                ])
            )
            edge_seq_list.append(
                np.concatenate([
                    np.asarray([0],dtype=np.int64),
                    selected_edges
                ])
            )
            ts_seq_list.append(
                np.concatenate([
                    np.asarray([0.0],dtype=np.float32),
                    (cut_time-selected_times).astype(np.float32)
                ])
            )
        return {
            "node": node_seq_list,
            "edge": edge_seq_list,
            "ts": ts_seq_list
        }
