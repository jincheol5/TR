import pandas as pd
import numpy as np
import torch
from .temporal_graph import TemporalGraph

class TGN_Graph(TemporalGraph):
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
            bipartite=bipartite
        )
        # set node_ft, edge_ft, node_dim, edge_dim
        if node_ft is None: 
            self.set_node_ft(node_dim=node_dim)
        else: 
            self.set_node_ft(node_ft=node_ft)
        if edge_ft is None: 
            self.set_edge_ft(edge_dim=edge_dim)
        else: 
            self.set_edge_ft(edge_ft=edge_ft)
        self.node_dim=node_dim
        self.edge_dim=edge_dim

    def set_node_ft(self,
            node_ft:np.ndarray=None,
            node_dim:int=32
        ):
        """
        Input:
            node_ft_np: 
                np.ndarray of shape (n_node+1, node_dim) 또는 None.
                0번째 행은 padding node feature (zero vector).
            node_dim: int
        """
        if node_ft is None:
            self.node_dim=node_dim
            self.node_ft=torch.zeros(
                (self.n_node+1,node_dim),
                dtype=torch.float32
            )
        else:
            self.node_dim=node_ft.shape[1]
            self.node_ft=torch.as_tensor(
                node_ft,
                dtype=torch.float32
            )

    def set_edge_ft(self,
            edge_ft:np.ndarray=None,
            edge_dim:int=32
        ):
        """
        Input:
            edge_ft_np: 
                np.ndarray of shape (n_edge+1, edge_dim)
                0번째 행은 padding edge의 feature(보통 zero vector).
                None이면 zero feature 생성.
            edge_dim: int
        """
        if edge_ft is None:
            self.edge_dim=edge_dim
            self.edge_ft=torch.zeros(
                (self.n_event+1,edge_dim),
                dtype=torch.float32
            )
        else:
            self.edge_dim=edge_ft.shape[1]
            self.edge_ft=torch.as_tensor(
                edge_ft,
                dtype=torch.float32
            )

    def get_node_ft(self,node:torch.Tensor=None):
        """
        Input:
            node: [B,]
        Return:
            node_ft
        """
        device=node.device
        return self.node_ft.to(device=device)[node]

    def get_edge_ft(self,edge:torch.Tensor=None):
        """
        Input:
            edge: [B,]
        Return:
            edge_ft
        """
        device=edge.device
        return self.edge_ft.to(device=device)[edge]

    def get_temporal_neighbor(self,
            tar:torch.Tensor,
            tar_t:torch.Tensor,
            n_neighbor:int
        ):
        """
        Input:
            tar: [n_tar,]
            tar_t: [n_tar,]
            n_neighbor: int
        Return:
            neighbor: [n_tar,n_neighbor]
            neighbor_t: [n_tar,n_neighbor]
            neighbor_ts: [n_tar,n_neighbor]
            neighbor_edge: [n_tar,n_neighbor]
        """
        device=tar.device
        n_tar=tar.size(0)
        neighbor=torch.zeros(
            (n_tar,n_neighbor),
            dtype=torch.long,
            device=device
        )
        neighbor_t=torch.zeros(
            (n_tar,n_neighbor),
            dtype=torch.float,
            device=device
        )
        neighbor_ts=torch.zeros(
            (n_tar,n_neighbor),
            dtype=torch.float,
            device=device
        )
        neighbor_edge=torch.zeros(
            (n_tar,n_neighbor),
            dtype=torch.long,
            device=device
        )
        for i in range(n_tar):
            tar_id=int(tar[i].item())
            cut_time=float(tar_t[i].item())

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
                neighbor[i,offset+idx]=src
                neighbor_t[i,offset+idx]=t
                neighbor_ts[i,offset+idx]=abs(cut_time-t)
                neighbor_edge[i,offset+idx]=e_id

        # compute neighbor_mask
        neighbor_mask=(neighbor!=0) # [B,n_neighbor], bool
        return {
            "neighbor":neighbor, 
            "neighbor_mask":neighbor_mask,
            "neighbor_t":neighbor_t, 
            "neighbor_ts":neighbor_ts, 
            "neighbor_edge":neighbor_edge
        }