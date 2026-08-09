import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from .TGN_graph import TGN_Graph

class DyGFormer_Graph(TGN_Graph):
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

    """
    << STEP >>
    1. src와 dst에 대한 S_src, S_dst 계산
    2. S_node로부터 X_n, X_e, X_t 계산
    3. S_src와 S_dst로부터 C_src, C_dst 구한 후 X_src_c, X_dst_c 계산
    4. node의 X_n, X_e, X_t, X_c를 각각 patching 후 M_n, M_e, M_t, M_c 계산
    """

    def get_history_seq(self,
            node:torch.Tensor,
            event_t:torch.Tensor,
            max_n_neighbor:int
        ):
        """
        batch 내 각 노드의 1-hop history sequence를 구해 list로 반환.
        각 sequence에 자기 자신의 정보를 맨 앞에 추가한다.
        이때, edge_id는 0, timespan은 0.0으로 추가한다.

        Input:
            node: [batch_size,]
            event_t: [batch_size,]
            max_n_neighbor: int
        Return:
            node_seq_list: list of each node's history node sequence 
            edge_seq_list: list of each node's history edge sequence
            ts_seq_list: list of each node's history timespan sequence
        """
        node_seq_list=[]
        edge_seq_list=[]
        ts_seq_list=[]
        for node_id,cut_time in zip(
                node.detach().cpu().numpy(),
                event_t.detach().cpu().numpy()
            ):
            neighbors=self.adj.get(int(node_id),[])
            times=np.asarray(
                self.adj_t.get(int(node_id),[]),
                dtype=np.float32
            )
            cut_idx=np.searchsorted(
                times,
                cut_time,
                side="left"
            )
            start_idx=max(0,cut_idx-max_n_neighbor)
            selected_neighbors=neighbors[start_idx:cut_idx]
            selected_times=times[start_idx:cut_idx]
            node_seq_list.append(
                np.asarray(
                    [int(node_id)] +
                    [n for n,_ in selected_neighbors],
                    dtype=np.int64
                )
            )
            edge_seq_list.append(
                np.asarray(
                    [0] +
                    [e for _,e in selected_neighbors],
                    dtype=np.int64
                )
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

    def get_padded_seq_vec(self,
            node_seq_list:list,
            edge_seq_list:list,
            ts_seq_list:list,
            device:torch.device
        ):
        """
        Input:
            node_seq_list: list of each node's history node sequence 
            edge_seq_list: list of each node's history edge sequence
            ts_seq_list: list of each node's history timespan sequence
        Return:
            node_seq_vec: [batch_size,max_seq_len,node_dim] 
            edge_seq_vec: [batch_size,max_seq_len,edge_dim]
            ts_seq_vec: [batch_size,max_seq_len,1]
        """
        batch_size=len(node_seq_list)
        max_seq_len=max(
            (len(seq) for seq in node_seq_list),
            default=0
        )
        node_seq=torch.zeros(
            (batch_size,max_seq_len),
            dtype=torch.long,
            device=device
        )
        edge_seq=torch.zeros(
            (batch_size,max_seq_len),
            dtype=torch.long,
            device=device
        )
        ts_seq=torch.zeros(
            (batch_size,max_seq_len),
            dtype=torch.float,
            device=device
        )
        for i in range(batch_size):
            seq_len=len(node_seq_list[i])
            node_seq[i,:seq_len]=torch.as_tensor(
                node_seq_list[i],device=device
            )
            edge_seq[i,:seq_len]=torch.as_tensor(
                edge_seq_list[i],device=device
            )
            ts_seq[i,:seq_len]=torch.as_tensor(
                ts_seq_list[i],device=device
            )
        return {
            "node": node_seq, # [batch_size,max_seq_len]
            "node_ft": self.get_node_ft(node_seq), # [batch_size,max_seq_len,node_dim] 
            "edge_ft": self.get_edge_ft(edge_seq), # [batch_size,max_seq_len,edge_dim] 
            "ts": ts_seq.unsqueeze(-1) # [batch_size,max_seq_len,1] 
        }

    def get_co_occurrence_vec(self,
            src_seq:torch.Tensor,
            dst_seq:torch.Tensor
        ):
        """
        Compute Neighbor Co-occurrence Vector.

        node==0일 때 continue하기 때문에 padding node의 co-occurrence는 항상 [0,0]으로 유지한다.

        Input:
            src_seq: [batch_size,max_src_seq_len,]
            dst_seq: [batch_size,max_dst_seq_len,]
        Return: 
            src_co_vec: [batch_size,max_src_seq_len,2]
            dst_co_vec: [batch_size,max_dst_seq_len,2] 
        """
        device=src_seq.device
        batch_size=src_seq.size(0)
        src_co_vec=torch.zeros(
            (*src_seq.shape,2),
            dtype=torch.float32,
            device=device
        )
        dst_co_vec=torch.zeros(
            (*dst_seq.shape,2),
            dtype=torch.float32,
            device=device
        )
        for b in range(batch_size):
            each_src_seq=src_seq[b]
            each_dst_seq=dst_seq[b]
            for idx,node in enumerate(each_src_seq):
                if node==0:
                    continue
                src_co_vec[b,idx,0]=(each_src_seq==node).sum()
                src_co_vec[b,idx,1]=(each_dst_seq==node).sum()
            for idx,node in enumerate(each_dst_seq):
                if node==0:
                    continue
                dst_co_vec[b,idx,0]=(each_src_seq==node).sum()
                dst_co_vec[b,idx,1]=(each_dst_seq==node).sum()
        return {
            "src": src_co_vec, # [batch_size,max_src_seq_len,2]
            "dst": dst_co_vec # [batch_size,max_dst_seq_len,2] 
        }

    def get_patching_vec(self,
            node_seq_ft:torch.Tensor,
            edge_seq_ft:torch.Tensor,
            ts_seq_ft:torch.Tensor,
            co_seq_ft:torch.Tensor,
            patch_size:int
        ):
        """
        Patching Technique in DyGFormer
        p=patch_size
        l=max_seq_len/patch_size

        Input:
            node_seq_ft: [batch_size,max_seq_len,node_dim]
            edge_seq_ft: [batch_size,max_seq_len,edge_dim]
            ts_seq_ft: [batch_size,max_seq_len,time_dim]
            co_seq_ft:[batch_size,max_seq_len,co_dim]
            patch_size: int
        Return:
            M_n: [B,l,node_dim x p]
            M_e: [B,l,edge_dim x p]
            M_t: [B,l,time_dim x p]
            M_c: [B,l,co_dim x p]
        """
        batch_size,seq_len,_=node_seq_ft.shape

        # patch_size의 배수가 되도록 padding
        pad_len=(patch_size-seq_len%patch_size)%patch_size
        if pad_len>0:
            node_seq_ft=F.pad(
                node_seq_ft,
                (0,0,0,pad_len) # seq_len 차원에 pad_len만큼 zero padding 추가
            )
            edge_seq_ft=F.pad(
                edge_seq_ft,
                (0,0,0,pad_len)
            )
            ts_seq_ft=torch.nn.functional.pad(
                ts_seq_ft, 
                (0,0,0,pad_len)
            )
            co_seq_ft=torch.nn.functional.pad(
                co_seq_ft, 
                (0,0,0,pad_len)
            )
        seq_len=node_seq_ft.size(1)
        l=seq_len//patch_size
        return {
            "node": node_seq_ft.reshape(batch_size,l,-1), # M_n: [B,l,node_dim x p]
            "edge": edge_seq_ft.reshape(batch_size,l,-1), # M_e: [B,l,edge_dim x p]
            "ts": ts_seq_ft.reshape(batch_size,l,-1), # M_t: [B,l,time_dim x p]
            "co": co_seq_ft.reshape(batch_size,l,-1) # M_c: [B,l,co_dim x p]
        }
