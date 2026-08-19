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
            device:torch.device
        ):
        """
        Input:
            node_seq_list: list of each node's history node sequence 
            edge_seq_list: list of each node's history edge sequence
            ts_seq_list: list of each node's history timespan sequence
        Return:
            node_seq: [B,max_seq_len] 
            node_seq_vec: [B,max_seq_len,node_dim] 
            edge_seq_vec: [B,max_seq_len,edge_dim]
            ts_seq_vec: [B,max_seq_len,1]
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
            "node_seq": node_seq, # [batch_size,max_seq_len]
            "ts_seq": ts_seq.unsqueeze(-1), # [batch_size,max_seq_len]
            "node_seq_vec": self.node_ft[node_seq], # [batch_size,max_seq_len,node_dim] 
            "edge_seq_vec": self.edge_ft[edge_seq] # [batch_size,max_seq_len,edge_dim] 
        }

    def get_co_occurrence_vec(self,
            src_seq:torch.Tensor,
            dst_seq:torch.Tensor
        ):
        """
        Compute Neighbor Co-occurrence Vector.

        node = 0일 때 continue하기 때문에 padding node의 co-occurrence는 항상 [0,0]으로 유지한다.

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
            "src_co_vec": src_co_vec, # [batch_size,max_src_seq_len,2]
            "dst_co_vec": dst_co_vec # [batch_size,max_dst_seq_len,2] 
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