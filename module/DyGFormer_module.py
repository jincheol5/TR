import torch
import torch.nn.functional as F

class DyGFormer_Module:
    def __init__(self,
            node_ft:torch.Tensor,
            edge_ft:torch.Tensor
        ):
        self.node_ft=node_ft
        self.edge_ft=edge_ft

    def get_padded_seq_vec(self,
            node_seq_list:list,
            edge_seq_list:list,
            ts_seq_list:list,
            max_seq_len:int,
            device:torch.device
        ):
        """
        CPU에서 수행

        Input:
            node_seq_list: list of each node's history node sequence 
            edge_seq_list: list of each node's history edge sequence
            ts_seq_list: list of each node's history timespan sequence
            max_seq_len: src/dst에 공통으로 적용할 최대 sequence 길이
        Return:
            node_seq: [B,max_seq_len] 
            node_seq_vec: [B,max_seq_len,node_dim] 
            edge_seq_vec: [B,max_seq_len,edge_dim]
            ts_seq_vec: [B,max_seq_len,1]
        """
        batch_size=len(node_seq_list)
        node_seq=torch.zeros(
            (batch_size,max_seq_len),
            dtype=torch.long
        )
        edge_seq=torch.zeros(
            (batch_size,max_seq_len),
            dtype=torch.long
        )
        ts_seq=torch.zeros(
            (batch_size,max_seq_len),
            dtype=torch.float
        )
        for i in range(batch_size):
            seq_len=len(node_seq_list[i])
            node_seq[i,:seq_len]=torch.as_tensor(node_seq_list[i])
            edge_seq[i,:seq_len]=torch.as_tensor(edge_seq_list[i])
            ts_seq[i,:seq_len]=torch.as_tensor(ts_seq_list[i])
        return {
            "node_seq": node_seq, # [batch_size,max_seq_len]
            "ts_seq": ts_seq.unsqueeze(-1), # [batch_size,max_seq_len]
            "node_seq_vec": self.node_ft[node_seq], # [batch_size,max_seq_len,node_dim]
            "edge_seq_vec": self.edge_ft[edge_seq] # [batch_size,max_seq_len,edge_dim]
        }

    def get_co_occurrence_vec(
            self,
            src_seq:torch.Tensor,
            dst_seq:torch.Tensor
        ):
        """
        Compute Neighbor Co-occurrence Vector.
        CPU에서 수행

        torch.bincount(): 0 이상의 정수 텐서에서 각 정수가 몇 번 등장했는지 세는 함수
        
        Input:
            src_seq: [B, src_len]
            dst_seq: [B, dst_len]

        Return:
            src_co_vec: [B, src_len, 2]
            dst_co_vec: [B, dst_len, 2]
        """
        src_co_vec=[]
        dst_co_vec=[]
        for src,dst in zip(src_seq, dst_seq):
            max_id=torch.max(src.max(),dst.max()).item()+1
            src_count=torch.bincount(src,minlength=max_id)
            dst_count=torch.bincount(dst,minlength=max_id)

            src_co_vec.append(
                torch.stack([
                    src_count[src],
                    dst_count[src]
                ], dim=-1)
            )
            dst_co_vec.append(
                torch.stack([
                    src_count[dst],
                    dst_count[dst]
                ], dim=-1)
            )

        src_co_vec=torch.stack(src_co_vec).float()
        dst_co_vec=torch.stack(dst_co_vec).float()

        # padding node
        src_co_vec[src_seq==0]=0
        dst_co_vec[dst_seq==0]=0
        return {
            "src_co_vec": src_co_vec,
            "dst_co_vec": dst_co_vec
        }

    def get_patching_vec(self,
            node_seq_vec:torch.Tensor,
            edge_seq_vec:torch.Tensor,
            ts_seq_vec:torch.Tensor,
            co_seq_vec:torch.Tensor,
            patch_size:int
        ):
        """
        Patching Technique in DyGFormer
        p=patch_size
        l=max_seq_len/patch_size

        Input:
            node_seq_vec: [batch_size,max_seq_len,node_dim]
            edge_seq_vec: [batch_size,max_seq_len,edge_dim]
            ts_seq_vec: [batch_size,max_seq_len,time_dim]
            co_seq_vec:[batch_size,max_seq_len,co_dim]
            patch_size: int
        Return:
            M_n: [B,l,node_dim x p]
            M_e: [B,l,edge_dim x p]
            M_t: [B,l,time_dim x p]
            M_c: [B,l,co_dim x p]
        """
        batch_size,seq_len,_=node_seq_vec.shape

        # patch_size의 배수가 되도록 padding
        pad_len=(patch_size-seq_len%patch_size)%patch_size
        if pad_len>0:
            node_seq_vec=F.pad(
                node_seq_vec,
                (0,0,0,pad_len) # seq_len 차원에 pad_len만큼 zero padding 추가
            )
            edge_seq_vec=F.pad(
                edge_seq_vec,
                (0,0,0,pad_len)
            )
            ts_seq_vec=torch.nn.functional.pad(
                ts_seq_vec, 
                (0,0,0,pad_len)
            )
            co_seq_vec=torch.nn.functional.pad(
                co_seq_vec, 
                (0,0,0,pad_len)
            )
        seq_len=node_seq_vec.size(1)
        l=seq_len//patch_size
        return {
            "node": node_seq_vec.reshape(batch_size,l,-1), # M_n: [B,l,node_dim x p]
            "edge": edge_seq_vec.reshape(batch_size,l,-1), # M_e: [B,l,edge_dim x p]
            "ts": ts_seq_vec.reshape(batch_size,l,-1), # M_t: [B,l,time_dim x p]
            "co": co_seq_vec.reshape(batch_size,l,-1) # M_c: [B,l,co_dim x p]
        }
