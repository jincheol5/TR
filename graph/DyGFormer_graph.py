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

    def get_historical_seq(self,
            node:torch.Tensor,
            event_t:torch.Tensor,
            max_n_neighbor:int
        ):
        """
        Input:
            node: [N,]
            event_t: [N,]
        Return:
            seq_node: list of [history_len] arr
            seq_edge: list of [history_len] arr
            seq_ts: list of [history_len] arr
        """
        seq_node=[]
        seq_edge=[]
        seq_ts=[]
        node_np=node.detach().cpu().numpy()
        event_t_np=event_t.detach().cpu().numpy()

        for node_id,cut_time in zip(node_np,event_t_np):
            node_id=int(node_id)
            cut_time=float(cut_time)

            neighbors=self.adj.get(node_id,[]) # list of (src,edge_id)
            times=self.adj_t.get(node_id,[])

            if len(neighbors)==0:
                seq_node.append(np.array([],dtype=np.longlong))
                seq_edge.append(np.array([],dtype=np.longlong))
                seq_ts.append(np.array([],dtype=np.float32))
                continue

            times_np=np.asarray(times,dtype=np.float32)

            # t < cut_time 인 interaction만 선택
            cut_idx=np.searchsorted(
                times_np,
                cut_time,
                side="left"
            )

            selected_neighbors=neighbors[:cut_idx]
            selected_times=times_np[:cut_idx]
            history_nodes=np.array(
                [src for src,_ in selected_neighbors],
                dtype=np.longlong
            )
            history_edges=np.array(
                [edge_id for _,edge_id in selected_neighbors],
                dtype=np.longlong
            )
            history_ts=np.abs(
                cut_time-selected_times
            ).astype(np.float32)

            if len(history_nodes)>max_n_neighbor: # 최신 시퀀스만 가져옴
                history_nodes=history_nodes[-max_n_neighbor:]
                history_edges=history_edges[-max_n_neighbor:]
                history_ts=history_ts[-max_n_neighbor:]

            seq_node.append(history_nodes)
            seq_edge.append(history_edges)
            seq_ts.append(history_ts)
        return {
            "seq_node": seq_node,
            "seq_edge": seq_edge,
            "seq_ts": seq_ts
        }
