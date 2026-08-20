import torch
import torch.nn as nn
from graph import DyGFormer_Graph
from module import TimeEncoder,TransformerEncoderBlock,DyGFormer_Module

class DyGFormer(nn.Module):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            latent_dim:int,
            time_dim:int,
            output_dim:int,
            co_dim:int,
            common_dim:int,
            patch_size:int,
            graph:DyGFormer_Graph,
            n_neighbor:int,
            n_layer:int,
            n_head:int
        ):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.latent_dim=latent_dim
        self.time_dim=time_dim
        self.output_dim=output_dim
        self.co_dim=co_dim
        self.common_dim=common_dim
        self.patch_size=patch_size
        self.graph=graph
        self.n_neighbor=n_neighbor
        self.n_layer=n_layer
        self.n_head=n_head

        # DyGFormer Module
        self.module=DyGFormer_Module(
            node_ft=self.graph.get_node_ft(),
            edge_ft=self.graph.get_edge_ft()
        )

        # time encoder
        self.time_encoder=TimeEncoder(time_dim=time_dim)

        # Neighbor Co-occurrence Encoding
        self.NCoE=nn.Sequential(
            nn.Linear(in_features=1,out_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim,out_features=co_dim)
        )

        # patch_encoder
        self.patch_encoder=nn.ModuleDict(
            {
                "node":nn.Linear(
                    in_features=patch_size*node_dim,
                    out_features=common_dim
                ),
                "edge":nn.Linear(
                    in_features=patch_size*edge_dim,
                    out_features=common_dim
                ),
                "time":nn.Linear(
                    in_features=patch_size*time_dim,
                    out_features=common_dim
                ),
                "co":nn.Linear(
                    in_features=patch_size*co_dim,
                    out_features=common_dim
                )
            }
        )

        # transformer encoder
        self.transformer_encoders=nn.ModuleList(
            [
                TransformerEncoderBlock(
                    attn_dim=4*common_dim,
                    latent_dim=latent_dim,
                    n_head=n_head
                )
                for _ in range(n_layer)
            ]
        )

        # Time-aware Node Representation
        self.output_layer=nn.Linear(
            in_features=4*common_dim,
            out_features=output_dim
        )
        # decoder
        self.decoder=nn.Sequential(
            nn.Linear(
                in_features=output_dim+output_dim,
                out_features=latent_dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=latent_dim,
                out_features=1
            )
        )

    def forward(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            query_t:torch.Tensor,
        ):
        """
        Input: sampling 된 pos/neg pair의 src,dst,query_time
            src: [B,] 
            dst: [B,]
            query_t: [B,]
        Return:
            pred_logit: [B,1]
        """
        batch_size=src.size(0) # B
        device=src.device

        ### 1. get sequence data
        src_result=self.graph.get_history_seq(
            node=src,
            event_t=query_t,
            n_neighbor=self.n_neighbor
        )
        src_node_seq_list=src_result["node"]
        src_edge_seq_list=src_result["edge"]
        src_ts_seq_list=src_result["ts"]

        dst_result=self.graph.get_history_seq(
            node=dst,
            event_t=query_t,
            n_neighbor=self.n_neighbor
        )
        dst_node_seq_list=dst_result["node"]
        dst_edge_seq_list=dst_result["edge"]
        dst_ts_seq_list=dst_result["ts"]

        ### 2. get padded sequence
        max_seq_len=max(
            max(
                len(seq)
                for seq in src_node_seq_list
            ),
            max(
                len(seq)
                for seq in dst_node_seq_list
            )
        )
        src_result=self.module.get_padded_seq_vec(
            node_seq_list=src_node_seq_list,
            edge_seq_list=src_edge_seq_list,
            ts_seq_list=src_ts_seq_list,
            max_seq_len=max_seq_len,
            device=device
        )
        src_seq=src_result["node_seq"]
        src_ts_seq=src_result["ts_seq"]
        src_node_seq_vec=src_result["node_seq_vec"]
        src_edge_seq_vec=src_result["edge_seq_vec"]

        dst_result=self.module.get_padded_seq_vec(
            node_seq_list=dst_node_seq_list,
            edge_seq_list=dst_edge_seq_list,
            ts_seq_list=dst_ts_seq_list,
            max_seq_len=max_seq_len,
            device=device
        )
        dst_seq=dst_result["node_seq"]
        dst_ts_seq=dst_result["ts_seq"]
        dst_node_seq_vec=dst_result["node_seq_vec"]
        dst_edge_seq_vec=dst_result["edge_seq_vec"]

        ### 3. compute co-occurrence vector
        co_result=self.module.get_co_occurrence_vec(
            src_seq=src_seq,
            dst_seq=dst_seq
        )
        src_co_vec=co_result["src_co_vec"]
        dst_co_vec=co_result["dst_co_vec"]

        ### 4. encode timespan 
        src_ts_seq_vec=self.time_encoder(src_ts_seq) 
        dst_ts_seq_vec=self.time_encoder(dst_ts_seq)

        ### 5. apply NCoE
        src_co_vec_0=src_co_vec[:,:,0:1] 
        src_co_vec_1=src_co_vec[:,:,1:2] 

        dst_co_vec_0=dst_co_vec[:,:,0:1] 
        dst_co_vec_1=dst_co_vec[:,:,1:2] 

        co_vec=torch.concat(
            [
                src_co_vec_0,
                src_co_vec_1,
                dst_co_vec_0,
                dst_co_vec_1
            ],
            dim=0
        )
        co_vec=self.NCoE(co_vec)
        src_co_vec_0,src_co_vec_1,dst_co_vec_0,dst_co_vec_1=torch.chunk(
            co_vec,
            chunks=4,
            dim=0
        )
        src_co_vec=src_co_vec_0+src_co_vec_1 # [B,max_seq_len,co_dim]
        dst_co_vec=dst_co_vec_0+dst_co_vec_1 # [B,max_seq_len,co_dim]

        ### 6. get patching sequence
        src_result=self.module.get_patching_vec(
            node_seq_vec=src_node_seq_vec,
            edge_seq_vec=src_edge_seq_vec,
            ts_seq_vec=src_ts_seq_vec,
            co_seq_vec=src_co_vec,
            patch_size=self.patch_size
        )
        src_M_n=src_result["node"]
        src_M_e=src_result["edge"]
        src_M_t=src_result["ts"]
        src_M_c=src_result["co"]

        dst_result=self.module.get_patching_vec(
            node_seq_vec=dst_node_seq_vec,
            edge_seq_vec=dst_edge_seq_vec,
            ts_seq_vec=dst_ts_seq_vec,
            co_seq_vec=dst_co_vec,
            patch_size=self.patch_size
        )
        dst_M_n=dst_result["node"]
        dst_M_e=dst_result["edge"]
        dst_M_t=dst_result["ts"]
        dst_M_c=dst_result["co"]

        ### 7. apply patch encoder
        M_n=torch.concat([src_M_n,dst_M_n],dim=0) # [2B,l,node_dim x p]
        M_e=torch.concat([src_M_e,dst_M_e],dim=0) # [2B,l,edge_dim x p]
        M_t=torch.concat([src_M_t,dst_M_t],dim=0) # [2B,l,time_dim x p]
        M_c=torch.concat([src_M_c,dst_M_c],dim=0) # [2B,l,co_dim x p]

        M_n=self.patch_encoder["node"](M_n) # [2B,l,patch_dim]
        M_e=self.patch_encoder["edge"](M_e) # [2B,l,patch_dim]
        M_t=self.patch_encoder["time"](M_t) # [2B,l,patch_dim]
        M_c=self.patch_encoder["co"](M_c) # [2B,l,patch_dim]

        src_M_n,dst_M_n=torch.chunk(M_n,chunks=2,dim=0) # [B,l,patch_dim]
        src_M_e,dst_M_e=torch.chunk(M_e,chunks=2,dim=0) # [B,l,patch_dim]
        src_M_t,dst_M_t=torch.chunk(M_t,chunks=2,dim=0) # [B,l,patch_dim]
        src_M_c,dst_M_c=torch.chunk(M_c,chunks=2,dim=0) # [B,l,patch_dim]

        ### 8. get updated node vec z
        src_z=torch.concat([src_M_n,src_M_e,src_M_t,src_M_c],dim=-1) # [B,l,4 x patch_dim]
        dst_z=torch.concat([dst_M_n,dst_M_e,dst_M_t,dst_M_c],dim=-1) # [B,l,4 x patch_dim]
        z=torch.concat([src_z,dst_z],dim=1) # [B,2l,4 x patch_dim]

        ### 9. apply transformer encoder
        for transformer_encoder in self.transformer_encoders:
            z=transformer_encoder(z)
        src_z,dst_z=torch.chunk(z,chunks=2,dim=1) # [B,l,4 x patch_dim]

        ### 10. get Time-aware Node Representation h
        src_h=src_z.mean(dim=1) # [B,4 x patch_dim]
        dst_h=dst_z.mean(dim=1) # [B,4 x patch_dim]
        h=torch.concat([src_h,dst_h],dim=0) # [2B,4 x patch_dim]
        h=self.output_layer(h) # [2B,output_dim]
        src_h,dst_h=torch.chunk(h,chunks=2,dim=0) # [B,output_dim]

        ### 11. predict TR
        pair_vec=torch.concat([src_h,dst_h],dim=-1) # [B,output_dim+output_dim]
        pred_logit=self.decoder(pair_vec) # [B,1]
        return pred_logit
