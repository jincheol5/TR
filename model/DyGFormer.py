import torch
import torch.nn as nn
from graph import DyGFormer_Graph
from module import TimeEncoder,TransformerEncoderBlock

class DyGFormer_Base(nn.Module):
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
            max_n_neighbor:int,
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
        self.max_n_neighbor=max_n_neighbor
        self.n_layer=n_layer
        self.n_head=n_head

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

    def forward(self):
        return NotImplemented

class DyGFormer_TR(DyGFormer_Base):
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
            max_n_neighbor:int,
            n_layer:int,
            n_head:int
        ):
        super(DyGFormer_TR,self).__init__(
            node_dim=node_dim,
            edge_dim=edge_dim,
            latent_dim=latent_dim,
            time_dim=time_dim,
            output_dim=output_dim,
            co_dim=co_dim,
            common_dim=common_dim,
            patch_size=patch_size,
            graph=graph,
            max_n_neighbor=max_n_neighbor,
            n_layer=n_layer,
            n_head=n_head
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
            pos_pair:dict,
            neg_pair:dict
        ):
        """
        Input:
            pos_pair: dict
                key: src, dst
                value: 
                    src: [B,] 
                    dst: [B,] 
            neg_pair: dict
                key: src, dst
                value: 
                    src: [B,] 
                    dst: [B,] 
        """
        ### 0. unpack node pair dict
        pos_src=pos_pair["src"]
        pos_dst=pos_pair["dst"]
        neg_src=neg_pair["src"]
        neg_dst=neg_pair["dst"]
        src=torch.concat([pos_src,neg_src],dim=0) # [2B,]
        dst=torch.concat([pos_dst,neg_dst],dim=0) # [2B,]