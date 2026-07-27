import torch
import torch.nn as nn
from typing_extensions import Literal
from graph import TemporalGraph
from .time_encoder import TimeEncoder

class Memory(nn.Module):
    def __init__(self,
            n_node:int,
            mem_dim:int=32
        ):
        super().__init__()
        self.n_node=n_node
        self.mem_dim=mem_dim
        self.mem_ft=nn.Parameter(
            torch.zeros((n_node+1,mem_dim)),
            requires_grad=False
        )
        self.last_update_t=nn.Parameter(
            torch.zeros(n_node+1),
            requires_grad=False
        )
    
    def init_memory_state(self):
        """
        initialize all the mem_ft and last_update_t to zero vectors, which should be called at the start of each epoch
        """
        self.mem_ft.data.zero_()
        self.last_update_t.data.zero_()

    def get_mem_ft(self,
            node:torch.Tensor
        ):
        """
        Input:
            node: [N,]
        Output:
            mem_ft: [N,mem_dim]
        """
        return self.mem_ft[node]

    def get_node_timespan(self,
            node:torch.Tensor,
            event_t:torch.Tensor
        ):
        """
        Input:
            node: [N,]
            event_t: [N,]
        Return:
            node_ts: [N,1]
        """
        node_ts=torch.abs(
            event_t-self.last_update_t[node]
        )
        return node_ts.unsqueeze(-1) # [N,1]

    def update_memory_state(self,
            node:torch.Tensor,
            mem_ft:torch.Tensor,
            event_t:torch.Tensor
        ):
        """
        Batch 내의 N개의 node들의 새로운 mem_ft와 last_update_t 업데이트 
        Input:
            node: [N,]
            mem_ft: [N,mem_dim]
            event_t: [N,]
        """
        self.mem_ft[node]=mem_ft
        self.last_update_t[node]=event_t

    def detach_memory_state(self):
        """
        """
        self.mem_ft.detach_()

class MemoryUpdater(nn.Module):
    def __init__(self,
            mem_dim:int,
            edge_dim:int,
            msg_dim:int,
            time_dim:int,
            time_encoder:TimeEncoder,
            graph:TemporalGraph,
            memory:Memory,
            msg_fn:Literal["concat","mlp"]="concat",
            aggr_fn:Literal["last","mean"]="last"
        ):
        super().__init__()
        self.mem_dim=mem_dim
        self.edge_dim=edge_dim
        self.msg_dim=msg_dim
        self.time_dim=time_dim
        self.msg_fn=msg_fn
        self.aggr_fn=aggr_fn

        # data
        self.graph=graph
        self.memory=memory

        # module
        self.time_encoder=time_encoder
        if msg_fn=="mlp":
            self.src_mlp=nn.Sequential(
                nn.Linear(
                    in_features=mem_dim+mem_dim+time_dim+edge_dim,
                    out_features=msg_dim
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features=msg_dim,
                    out_features=msg_dim
                )
            )
            self.dst_mlp=nn.Sequential(
                nn.Linear(
                    in_features=mem_dim+mem_dim+time_dim+edge_dim,
                    out_features=msg_dim
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features=msg_dim,
                    out_features=msg_dim
                )
            )
    
    def create_message(self,
            src,
            dst,
            edge,
            event_t
        ):
        """
        Input:
            src: [B,]
            dst: [B,]
            edge: [B,]
            event_t: [B,]
        Output:
            src_msg: [B,msg_dim]
            dst_msg: [B,msg_dim]
        """
        src_mem=self.memory.get_mem_ft(node=src)
        src_ts=self.memory.get_node_timespan(node=src,event_t=event_t)
        src_ts_ft=self.time_encoder(src_ts)

        dst_mem=self.memory.get_mem_ft(node=dst)
        dst_ts=self.memory.get_node_timespan(node=dst,event_t=event_t)
        dst_ts_ft=self.time_encoder(dst_ts)

        edge_ft=self.graph.get_edge_ft(edge=edge)

        src_msg=torch.concat(
            [
                src_mem,
                dst_mem,
                src_ts_ft,
                edge_ft
            ],
            dim=-1
        ) # [B,mem_dim+mem_dim+time_dim+edge_dim]

        dst_msg=torch.concat(
            [
                dst_mem,
                src_mem,
                dst_ts_ft,
                edge_ft
            ],
            dim=-1
        ) # [B,mem_dim+mem_dim+time_dim,edge_dim]

        if self.msg_fn=="mlp":
            src_msg=self.src_mlp(src_msg) # [B,msg_dim]
            dst_msg=self.dst_mlp(dst_msg) # [B,msg_dim]
        return src_msg,dst_msg

    def aggregate_message(self,
            src,
            dst,
            src_msg,
            dst_msg,
            event_t
        ):
        """
        Message Aggregation

        Input:
            src: [B,]
            dst: [B,]
            src_msg: [B,msg_dim]
            dst_msg: [B,msg_dim]
            event_t: [B,]
        Output:
            aggr_node: [unique_N,]
            aggr_msg: [unique_N,msg_dim]
            aggr_event_t: [unique_N,]
        """
        device=src.device

        nodes=torch.concat([src,dst],dim=0) # [2B,]
        msgs=torch.concat([src_msg,dst_msg],dim=0) # [2B,msg_dim]
        times=torch.concat([event_t,event_t],dim=0) # [2B,]

        # event time 순으로 오름차순 정렬
        sorted_idx=torch.argsort(times)
        nodes=nodes[sorted_idx]
        msgs=msgs[sorted_idx]
        times=times[sorted_idx]

        msg_dict={}
        for node,msg,t in zip(nodes,msgs,times):
            node=node.item()
            if node not in msg_dict:
                msg_dict[node]=[]
            msg_dict[node].append((msg,t))

        aggr_node=[]
        aggr_msg=[]
        aggr_event_t=[]
        match self.aggr_fn:
            case "last":
                for node in msg_dict.keys():
                    last_msg,last_t=msg_dict[node][-1]
                    aggr_node.append(node)
                    aggr_msg.append(last_msg)
                    aggr_event_t.append(last_t)
            case "mean":
                for node in msg_dict.keys():
                    msg_list=[msg for msg,_ in msg_dict[node]]
                    last_t=msg_dict[node][-1][1]
                    aggr_node.append(node)
                    aggr_msg.append(
                        torch.mean(
                            torch.stack(msg_list,dim=0),
                            dim=0,
                        )
                    )
                    # interaction time은 가장 최근 시각 사용
                    aggr_event_t.append(last_t)
        aggr_node=torch.tensor(aggr_node,device=device) # [unique_N,]
        aggr_msg=torch.stack(aggr_msg,dim=0) # [unique_N,msg_dim]
        aggr_event_t=torch.stack(aggr_event_t,dim=0) # [unique_N,]
        return aggr_node,aggr_msg,aggr_event_t
    
    def update_memory_implement(self,aggr_node,aggr_msg,aggr_event_t):
        return NotImplemented

    def update_memory(self,
            src,
            dst,
            edge,
            event_t
        ):
        """
        자식 class의 update_memory 실행으로 자식 class에서 구현된 update_memory_implement 호출
        
        Input:
            src: [B,]
            tar: [B,]
            edge: [B,]
            event_t: [B,]
        """
        src_msg,dst_msg=self.create_message(
            src=src,
            dst=dst,
            edge=edge,
            event_t=event_t
        )
        aggr_node,aggr_msg,aggr_event_t=self.aggregate_message(
            src=src,
            dst=dst,
            src_msg=src_msg,
            dst_msg=dst_msg,
            event_t=event_t
        )
        updated_mem_ft=self.update_memory_implement(
            aggr_node=aggr_node,
            aggr_msg=aggr_msg,
            aggr_event_t=aggr_event_t
        )
        return updated_mem_ft # [unique_N,mem_dim]

class GRUMemoryUpdater(MemoryUpdater):
    def __init__(self,
            mem_dim:int,
            edge_dim:int,
            msg_dim:int,
            time_dim:int,
            time_encoder:TimeEncoder,
            memory:Memory,
            msg_fn:Literal["concat","mlp"]="concat",
            aggr_fn:Literal["last","mean"]="last"
        ):
        super(GRUMemoryUpdater,self).__init__(
            mem_dim=mem_dim,
            edge_dim=edge_dim,
            msg_dim=msg_dim,
            time_dim=time_dim,
            time_encoder=time_encoder,
            memory=memory,
            msg_fn=msg_fn,
            aggr_fn=aggr_fn
        )
        if self.msg_fn=="concat":
            input_size=mem_dim+mem_dim+time_dim+edge_dim
        else:
            input_size=msg_dim
        self.memory_updater=nn.GRUCell(
            input_size=input_size,
            hidden_size=mem_dim
        )

    def update_memory_implement(self,aggr_node,aggr_msg,aggr_event_t):
        """
        Input:
            aggr_node: [unique_N,]
            aggr_msg: [unique_N,msg_dim]
            aggr_event_t: [unique_N,]
        """
        pre_mem_ft=self.memory.get_mem_ft(node=aggr_node) # [unique_N,mem_dim]
        updated_mem_ft=self.memory_updater(aggr_msg,pre_mem_ft) # [unique_N,mem_dim]
        self.memory.update_memory_state(
            node=aggr_node,
            mem_ft=updated_mem_ft,
            event_t=aggr_event_t
        )
        return updated_mem_ft # [unique_N,mem_dim]