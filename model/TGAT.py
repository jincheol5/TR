import torch
import torch.nn as nn
from graph import TGN_Graph
from module import TimeEncoder,GraphAttnEmbedding

class TGAT_Base(nn.Module):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            latent_dim:int,
            time_dim:int,
            output_dim:int,
            graph:TGN_Graph,
            n_layer:int,
            n_neighbor:int,
            n_head:int
        ):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.latent_dim=latent_dim
        self.time_dim=time_dim
        self.output_dim=output_dim
        self.graph=graph
        self.n_layer=n_layer
        self.n_neighbor=n_neighbor
        self.n_head=n_head

        # time encoder
        self.time_encoder=TimeEncoder(time_dim=time_dim)

        # encoder
        self.encoder=GraphAttnEmbedding(
            node_dim=node_dim,
            edge_dim=edge_dim,
            latent_dim=latent_dim,
            time_dim=time_dim,
            output_dim=output_dim,
            graph=self.graph,
            n_layer=n_layer,
            n_neighbor=n_neighbor,
            n_head=n_head,
            use_memory=False,
            time_encoder=self.time_encoder
        )
    def forward(self):
        return NotImplemented

class TGAT_TR(TGAT_Base):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            latent_dim:int,
            time_dim:int,
            output_dim:int,
            graph:TGN_Graph,
            n_layer:int,
            n_neighbor:int,
            n_head:int
        ):
        super(TGAT_TR,self).__init__(
            node_dim=node_dim,
            edge_dim=edge_dim,
            latent_dim=latent_dim,
            time_dim=time_dim,
            output_dim=output_dim,
            graph=graph,
            n_layer=n_layer,
            n_neighbor=n_neighbor,
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