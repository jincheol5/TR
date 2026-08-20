import math
import torch
import torch.nn as nn
from graph import TGN_Graph
from .time_encoder import TimeEncoder
from .mem_module import Memory
from .attn_module import TemporalGraphAttn

class EmbeddingModule(nn.Module):
    def __init__(self,
            node_dim:int=32,
            edge_dim:int=32,
            latent_dim:int=32,
            output_dim:int=32
        ):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.latent_dim=latent_dim
        self.output_dim=output_dim

    def compute_embedding(self):
        return NotImplemented

class IdentityEmbedding(EmbeddingModule):
    """
    memory state를 node embedding으로 직접 사용
    """
    def __init__(self,
            node_dim:int=32,
            edge_dim:int=32,
            mem_dim:int=32,
            latent_dim:int=32,
            output_dim:int=32,
            memory:Memory=None,
        ):
        super(IdentityEmbedding,self).__init__(
            node_dim=node_dim,
            edge_dim=edge_dim,
            latent_dim=latent_dim,
            output_dim=output_dim
        )
        self.memory=memory
        self.mem_dim=mem_dim
    def compute_embedding(self,
            tar:torch.Tensor
        ):
        return self.memory.get_mem_vec(node=tar) # [B,mem_dim]

class TimeProjectionEmbedding(EmbeddingModule):
    """
    JODIE-style time projection embedding
    emb(i,t)=(1+delta_t*w) x s^t_i
    """
    def __init__(self,
            node_dim:int=32,
            edge_dim:int=32,
            mem_dim:int=32,
            latent_dim:int=32,
            output_dim:int=32,
            memory:Memory=None,
        ):
        super(TimeProjectionEmbedding,self).__init__(
            node_dim=node_dim,
            edge_dim=edge_dim,
            latent_dim=latent_dim,
            output_dim=output_dim
        )
        self.memory=memory
        self.mem_dim=mem_dim
        # time-projection embedding layer
        class NormalLinear(nn.Linear):
            # From Jodie code
            def reset_parameters(self):
                stdv=1./math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0,stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0,stdv)
        self.embedding_layer=NormalLinear(in_features=1,out_features=self.mem_dim)

    def compute_embedding(self,
            tar:torch.Tensor,
            tar_t:torch.Tensor
        ):
        tar_mem=self.memory.get_mem_vec(node=tar) # [B,mem_dim]
        tar_ts=self.memory.get_node_timespan(
            node=tar,
            event_t=tar_t
        ) # [B,1]
        return tar_mem*(1+self.embedding_layer(tar_ts)) # [B,mem_dim]

class GraphEmbeddingModule(EmbeddingModule):
    def __init__(self,
            node_dim:int=32,
            edge_dim:int=32,
            mem_dim:int=32,
            latent_dim:int=32, 
            output_dim:int=32,
            time_dim:int=32,
            graph:TGN_Graph=None,
            memory:Memory=None,
            n_layer:int=1,
            n_neighbor:int=5,
            use_memory:bool=True,
            time_encoder:TimeEncoder=None
        ):
        super(GraphEmbeddingModule,self).__init__(
            node_dim=node_dim,
            edge_dim=edge_dim,
            latent_dim=latent_dim,
            output_dim=output_dim
        )
        self.time_dim=time_dim
        self.graph=graph
        self.n_layer=n_layer
        self.n_neighbor=n_neighbor
        self.use_memory=use_memory
        if use_memory:
            self.mem_dim=mem_dim
            self.memory=memory

        # module
        self.time_encoder=time_encoder

    def aggregate(self,
            tar_vec:torch.Tensor,
            tar_ts_vec:torch.Tensor,
            neighbor_vec:torch.Tensor,
            neighbor_ts_vec:torch.Tensor,
            neighbor_edge_ft:torch.Tensor,
            neighbor_mask:torch.Tensor,
            n_layer:int
        ):
        return NotImplemented

    def compute_embedding(self,
            tar:torch.Tensor,
            tar_t:torch.Tensor,
            n_layer:int=1
        ):
        """
        Input:
            tar: [n_tar,]
            tar_t: [n_tar,]
            n_layer: int
        Return:
            updated_tar_ft: [n_tar,output_dim]
        """
        tar_ft=self.graph.get_node_ft(node=tar) # [n_tar,node_dim]
        if self.use_memory:
            tar_mem=self.memory.get_mem_vec(node=tar)
            tar_ft=torch.concat(
                [tar_ft,tar_mem],
                dim=-1
            ) # [n_tar,node_dim+mem_dim]

        if n_layer==0:
            return tar_ft
        else:
            tar_vec=self.compute_embedding(
                tar=tar,
                tar_t=tar_t,
                n_layer=n_layer-1
            ) # [n_tar,tar_dim] if n_layer=1 else # [n_tar,output_dim]

            temporal_neighbor=self.graph.get_temporal_neighbor(
                tar=tar,
                tar_t=tar_t,
                n_neighbor=self.n_neighbor
            )
            neighbor=temporal_neighbor["neighbor"]
            neighbor_mask=temporal_neighbor["neighbor_mask"]
            neighbor_t=temporal_neighbor["neighbor_t"]
            neighbor_ts=temporal_neighbor["neighbor_ts"]
            neighbor_edge=temporal_neighbor["neighbor_edge"]

            # flatten for neighbor embedding
            n_tar,n_neighbor=neighbor.size()
            neighbor=neighbor.flatten() # [n_tar,n_neighbor] -> [n_tar x n_neighbor,]
            neighbor_t=neighbor_t.flatten() # [n_tar,n_neighbor] -> [n_tar x n_neighbor,]
            
            # apply time encoding
            tar_ts=torch.zeros_like(tar,dtype=torch.float32,device=tar.device).unsqueeze(-1) # [n_tar,1]
            tar_ts_vec=self.time_encoder(tar_ts) # [n_tar,time_dim]
            neighbor_ts=neighbor_ts.unsqueeze(-1) # [n_tar,n_neighbor,1]
            neighbor_ts_vec=self.time_encoder(neighbor_ts) # [n_tar,n_neighbor,time_dim]

            # get neighbor embedding
            neighbor_vec=self.compute_embedding(
                tar=neighbor,
                tar_t=neighbor_t,
                n_layer=n_layer-1
            ) # [n_tar x n_neighbor,tar_dim] if n_layer=1 else [n_tar x n_neighbor,output_dim] 

            # reshape
            neighbor_vec=neighbor_vec.reshape(n_tar,n_neighbor,-1) # -> [n_tar,n_neighbor,tar_dim] or [n_tar,n_neighbor,output_dim]

            # get edge feature
            neighbor_edge_ft=self.graph.get_edge_ft(edge=neighbor_edge) # [n_tar,n_neighbor,edge_dim]

            ### Aggregation
            updated_tar_vec=self.aggregate(
                tar_vec=tar_vec,
                tar_ts_vec=tar_ts_vec,
                neighbor_vec=neighbor_vec,
                neighbor_ts_vec=neighbor_ts_vec,
                neighbor_edge_ft=neighbor_edge_ft,
                neighbor_mask=neighbor_mask,
                n_layer=n_layer
            )
            return updated_tar_vec

class GraphSumEmbedding(GraphEmbeddingModule):
    def __init__(self,
            node_dim:int=32,
            edge_dim:int=32,
            mem_dim:int=32,
            latent_dim:int=32, 
            output_dim:int=32,
            time_dim:int=32,
            graph:TGN_Graph=None,
            memory:Memory=None,
            n_layer:int=1,
            n_neighbor:int=5,
            use_memory:bool=True,
            time_encoder:TimeEncoder=None
        ):
        super(GraphSumEmbedding,self).__init__(
            node_dim=node_dim,
            edge_dim=edge_dim,
            mem_dim=mem_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            time_dim=time_dim,
            graph=graph,
            memory=memory,
            n_layer=n_layer,
            n_neighbor=n_neighbor,
            use_memory=use_memory,
            time_encoder=time_encoder
        )
        # module
        input_dim=node_dim+mem_dim if self.use_memory else node_dim
        self.linear_1=torch.nn.ModuleList([
            nn.Linear(
                in_features=(input_dim if idx==0 else output_dim)+edge_dim+time_dim,
                out_features=output_dim
            )
            for idx in range(n_layer)
        ])
        self.linear_2=torch.nn.ModuleList([
            nn.Linear(
                in_features=(input_dim if idx==0 else output_dim)+time_dim+output_dim,
                out_features=output_dim
            )
            for idx in range(n_layer)
        ])
        self.relu=nn.ReLU()

    def aggregate(self,
            tar_vec:torch.Tensor,
            tar_ts_vec:torch.Tensor,
            neighbor_vec:torch.Tensor,
            neighbor_ts_vec:torch.Tensor,
            neighbor_edge_ft:torch.Tensor,
            neighbor_mask:torch.Tensor,
            n_layer:int
        ):
        """
        Input:
            tar_vec: [n_tar,tar_dim] or [n_tar,output_dim]
            tar_ts_vec: [n_tar,time_dim]
            neighbor_vec: [n_tar,n_neighbor,tar_dim] or [n_tar,n_neighbor,output_dim]
            neighbor_ts_vec: [n_tar,n_neighbor,time_dim]
            neighbor_edge_ft: [n_tar,n_neighbor,edge_dim]
            neighbor_mask: [n_tar,n_neighbor]
            n_layer: int
        """
        # neighbor feature 구성
        neighbor_vec=torch.concat(
            [neighbor_vec,neighbor_edge_ft,neighbor_ts_vec],
            dim=-1
        )

        # 각 neighbor message 변환: W_1(...)
        neighbor_msg=self.linear_1[n_layer-1](neighbor_vec) # [n_tar,n_neighbor,output_dim]

        # padding neighbor 제거, neighbor_mask가 valid=True라면 그대로 사용
        neighbor_msg=neighbor_msg.masked_fill(
            ~neighbor_mask.unsqueeze(-1),
            0.0
        )

        # sum aggregation
        neighbor_sum=torch.sum(neighbor_msg,dim=1)
        neighbor_sum=self.relu(neighbor_sum)

        # target node feature 구성
        tar_vec=torch.concat(
            [tar_vec,tar_ts_vec],
            dim=-1
        )  # [n_tar,tar_dim+time_dim] or [n_tar,output_dim+time_dim]

        # self feature와 neighbor aggregate 결합
        output=torch.concat(
            [tar_vec,neighbor_sum],
            dim=-1
        )  # [n_tar,tar_dim+time_dim+output_dim] or [n_tar,output_dim+time_dim+output_dim]

        # W_2(...)
        output=self.linear_2[n_layer-1](output) # [n_tar,output_dim]
        return output # [B,output_dim]

class GraphAttnEmbedding(GraphEmbeddingModule):
    def __init__(self,
            node_dim:int=32,
            edge_dim:int=32,
            mem_dim:int=32,
            latent_dim:int=32, 
            output_dim:int=32,
            time_dim:int=32,
            graph:TGN_Graph=None,
            memory:Memory=None,
            n_layer:int=1,
            n_neighbor:int=5,
            n_head:int=1,
            use_memory:bool=True,
            time_encoder:TimeEncoder=None
        ):
        super(GraphAttnEmbedding,self).__init__(
            node_dim=node_dim,
            edge_dim=edge_dim,
            mem_dim=mem_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            time_dim=time_dim,
            graph=graph,
            memory=memory,
            n_layer=n_layer,
            n_neighbor=n_neighbor,
            use_memory=use_memory,
            time_encoder=time_encoder
        )
        # module
        layer_0_input_dim=node_dim+mem_dim if self.use_memory else node_dim
        self.attn_layers=torch.nn.ModuleList([
                TemporalGraphAttn(
                    input_dim=layer_0_input_dim if idx==0 else output_dim,
                    edge_dim=edge_dim,
                    latent_dim=latent_dim,
                    output_dim=output_dim,
                    time_dim=time_dim,
                    n_head=n_head
                )
            for idx in range(n_layer)])
    def aggregate(self,
            tar_vec:torch.Tensor,
            tar_ts_vec:torch.Tensor,
            neighbor_vec:torch.Tensor,
            neighbor_ts_vec:torch.Tensor,
            neighbor_edge_ft:torch.Tensor,
            neighbor_mask:torch.Tensor,
            n_layer:int
        ):
        """
        Input:
            tar_vec: [n_tar,tar_dim] or [n_tar,output_dim]
            tar_ts_vec: [n_tar,time_dim]
            neighbor_vec: [n_tar,n_neighbor,tar_dim] or [n_tar,n_neighbor,output_dim]
            neighbor_ts_vec: [n_tar,n_neighbor,time_dim]
            neighbor_edge_ft: [n_tar,n_neighbor,edge_dim]
            neighbor_mask: [n_tar,n_neighbor]
            n_layer: int
        """
        aggr_module=self.attn_layers[n_layer-1]
        output=aggr_module(
            tar_vec=tar_vec,
            tar_ts_vec=tar_ts_vec,
            neighbor_vec=neighbor_vec,
            neighbor_ts_vec=neighbor_ts_vec,
            neighbor_edge_ft=neighbor_edge_ft,
            neighbor_mask=neighbor_mask
        ) # [n_tar,output_dim]
        return output 