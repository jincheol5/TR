import torch
import torch.nn as nn

class TemporalGraphAttn(nn.Module):
    """
    torch.nn.MultiheadAttention은 embed_dim % num_heads=0 이여야 함
    h^0_i=s^t_i||v^t_0
    """
    def __init__(self,
            input_dim:int,
            edge_dim:int,
            latent_dim:int,
            output_dim:int,
            time_dim:int,
            n_head:int=1
        ):
        super().__init__()
        self.input_dim=input_dim
        self.edge_dim=edge_dim
        self.latent_dim=latent_dim
        self.output_dim=output_dim
        self.time_dim=time_dim
        self.q_dim=input_dim+time_dim
        self.kv_dim=input_dim+edge_dim+time_dim
        self.multi_head_attn=nn.MultiheadAttention(
            embed_dim=self.q_dim,
            kdim=self.kv_dim,
            vdim=self.kv_dim,
            num_heads=n_head
        )
        self.MLPs=nn.Sequential(
            nn.Linear(
                in_features=self.q_dim+self.input_dim,
                out_features=self.latent_dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.latent_dim,
                out_features=self.output_dim
            )
        )
    def forward(self,
            tar_ft:torch.Tensor,
            tar_ts_ft:torch.Tensor,
            neighbor_ft:torch.Tensor,
            neighbor_ts_ft:torch.Tensor,
            neighbor_edge_ft:torch.Tensor,
            neighbor_mask:torch.Tensor
        ):
        """
        Input:
            tar_ft: [B,input_dim]
            tar_ts_ft: [B,time_dim]
            neighbor_ft: [B,K,input_dim]
            neighbor_ts_ft: [B,K,time_dim]
            neighbor_edge_ft: [B,K,edge_dim]
            neighbor_mask: [B,K]
        Output:
            updated tar_ft: # [B,output_dim]
        """
        ### set init
        tar_ft=tar_ft.unsqueeze(dim=1) # -> [B,1,input_dim]
        tar_ts_ft=tar_ts_ft.unsqueeze(dim=1) # -> [B,1,time_dim]

        query=torch.cat(
            [tar_ft,tar_ts_ft],
            dim=2
        ) # -> [B,1,input_dim+time_dim]
        key=torch.cat(
            [neighbor_ft,neighbor_edge_ft,neighbor_ts_ft],
            dim=2
        ) # -> [B,K,input_dim+edge_dim+time_dim]
        value=torch.cat(
            [neighbor_ft,neighbor_edge_ft,neighbor_ts_ft],
            dim=2
        ) # -> [B,K,input_dim+edge_dim+time_dim]

        ### set to [L,B,D]
        query=query.permute([1,0,2]) # -> [1,B,input_dim+time_dim] 
        key=key.permute([1,0,2]) # -> [N,B,input_dim+edge_dim+time_dim] 
        value=value.permute([1,0,2]) # -> [N,B,input_dim+edge_dim+time_dim] 

        ### transform n_mask for nn.MultiheadAttention's key_padding_mask
        # key_padding_mask에서는 True가 padding될 neighbor을 의미
        key_padding_mask=~neighbor_mask

        ### Compute mask of which target nodes have no valid neighbors
        # tensor.all() -> 모든 값이 true인지 검사하는 함수
        # 이웃이 하나도 없는 target node 의 경우, attn 수행을 위해  첫 번째 이웃 노드를 임시로 유효하게 수정 (fake neighbor)   
        # fake neighbor 에만 attn이 집중되도록 강제
        # 이후 처리 
        invalid_neighbor_mask=key_padding_mask.all(dim=1,keepdim=True) # [B,1], true=유효 neighbor 없음, false=유효 neighbor 존재
        key_padding_mask[invalid_neighbor_mask.squeeze(),0]=False 

        ### Multi-head attention
        attn_output,_=self.multi_head_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask
        ) # attn_output: [1,B,input_dim+time_dim], attn_weight: [B,1,N]
        attn_output=attn_output.squeeze() # -> [B,input_dim+time_dim]

        ### 이웃노드가 없는 target node의 attn 결과 feature를 0 tensor으로 후처리
        attn_output=attn_output.masked_fill(invalid_neighbor_mask,0) # mask_fill: mask=True인 위치를 value로 덮어쓰기

        ### MLPs
        tar_ft=tar_ft.squeeze() # -> [B,input_dim]
        ffn_input=torch.cat(
            [attn_output,tar_ft],
            dim=-1
        ) # -> [B,input_dim+time_dim||input_dim]
        output=self.MLPs(ffn_input) # [B,output_dim]
        return output

class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block in DyGFormer

    Block 구성:
        1. Multi-head Self-Attention (MSA) part
            1-1. Layer Normalization
            1-2. Multi-head Self-Attention
            1-3. Residual connection
        2. Feed-Forward Network (FFN) part
            2-1. Layer Normalization
            2-2. Feed-Forward Network
            2-3. Residual connection
    
    FFN 내부 비활성화 함수=GELU 사용
    """
    def __init__(self,
            attn_dim:int,
            latent_dim:int,
            n_head:int,
            dropout:float=0.1
        ):
        super().__init__()
        self.multi_head_attention=nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=n_head,
            dropout=dropout
        )
        self.dropout=nn.Dropout(dropout)
        self.linear_layers=nn.ModuleList([
            nn.Linear(in_features=attn_dim,out_features=latent_dim),
            nn.Linear(in_features=latent_dim,out_features=attn_dim)
        ])
        self.norm_layers=nn.ModuleList([
            nn.LayerNorm(attn_dim),
            nn.LayerNorm(attn_dim)
        ])
        self.gelu=nn.GELU()
    
    def forward(self,
            z:torch.Tensor
        ):
        """
        Input:
            z: [B,2l,4d], z=z_src에 z_dst을 dim=0으로 concat, 4d=attn_dim
        Output:
            out: [B,2l,4d]
        """
        ###### 1. MSA part
        ### Pre-LN
        norm_z=self.norm_layers[0](z) # [B,2l,4d]

        ### PyTorch MultiheadAttention 기본 입력 shape=[seq_len,batch_size,attn_dim]
        norm_z_t=norm_z.transpose(0,1) # [2l,B,4d] 

        ### Self-Attention: query = key = value = z
        attn_out,_=self.multi_head_attention(
            query=norm_z_t,
            key=norm_z_t,
            value=norm_z_t
        ) # [2l,B,4d]
        attn_out=attn_out.transpose(0,1)  # [B,2l,4d]

        ### Residual connection
        out=z+self.dropout(attn_out)  

        ###### 2. FFN part
        ### Pre-LN
        norm_out=self.norm_layers[1](out) # [B,2l,4d]

        ### FFN
        hidden=self.linear_layers[0](norm_out) # [B,2l,latent_dim]
        hidden=self.gelu(hidden)
        hidden=self.dropout(hidden)
        hidden=self.linear_layers[1](hidden) # [B,2l,4d]

        ### Residual connection
        out=out+self.dropout(hidden) # [B,2l,4d]
        return out