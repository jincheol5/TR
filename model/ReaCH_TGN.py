import torch
import torch.nn as nn
from typing_extensions import Literal
from graph import TGN_Graph
from module import TimeEncoder,Memory,GraphAttnEmbedding,MemoryUpdater,ReaCH_TGN_Module

class ReaCH_TGN(nn.Module):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            mem_dim:int,
            latent_dim:int,
            msg_dim:int,
            time_dim:int,
            output_dim:int,
            graph:TGN_Graph,
            n_layer:int,
            n_neighbor:int,
            n_head:int,
            msg_fn:Literal["concat","mlp"]="concat",
            aggr_fn:Literal["last","mean"]="last"
        ):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.mem_dim=mem_dim
        self.latent_dim=latent_dim
        self.msg_dim=msg_dim
        self.time_dim=time_dim
        self.output_dim=output_dim
        self.n_layer=n_layer
        self.n_neighbor=n_neighbor
        self.n_head=n_head
        self.msg_fn=msg_fn
        self.aggr_fn=aggr_fn
        
        # graph
        self.graph=graph

        # ReaCH-TGN Module
        self.module=ReaCH_TGN_Module() 

        # memory
        self.n_node=self.graph.get_num_node()
        self.memory=Memory(n_node=self.n_node,mem_dim=self.mem_dim)

        # time encoder
        self.time_encoder=TimeEncoder(time_dim=time_dim)

        # memory updater
        self.memory_updater=MemoryUpdater(
            mem_dim=mem_dim,
            edge_dim=edge_dim,
            msg_dim=msg_dim,
            time_dim=time_dim,
            time_encoder=self.time_encoder,
            graph=self.graph,
            memory=self.memory,
            msg_fn=msg_fn,
            aggr_fn=aggr_fn
        )

        # encoder
        self.encoder=GraphAttnEmbedding(
            node_dim=node_dim,
            edge_dim=edge_dim,
            mem_dim=mem_dim,
            latent_dim=latent_dim,
            time_dim=time_dim,
            output_dim=output_dim,
            graph=self.graph,
            memory=self.memory,
            n_layer=n_layer,
            n_neighbor=n_neighbor,
            n_head=n_head,
            use_memory=True,
            time_encoder=self.time_encoder
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


        ):
        """
        """
