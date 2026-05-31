import torch 
import torch.nn as nn
from typing_extensions import Literal
from .time_encoding import TimeEncoder
from .memory import Memory

class MemoryUpdater(nn.Module):
    def __init__(self,
            mem_dim:int,
            msg_dim:int,
            time_dim:int,
            time_encoder:TimeEncoder,
            memory:Memory,
            msg_fn:Literal["concat","mlp"]="concat",
            aggr_fn:Literal["last","mean"]="last"
        ):
        super().__init__()
        self.memory=memory
        self.msg_fn=msg_fn
        self.aggr_fn=aggr_fn
        self.mem_dim=mem_dim
        self.msg_dim=msg_dim
        self.time_dim=time_dim

        # set module
        self.time_encoder=time_encoder
        if msg_fn=="mlp":
            self.src_mlp=nn.Sequential(
                nn.Linear(in_features=mem_dim+mem_dim+time_dim,out_features=msg_dim),
                nn.ReLU(),
                nn.Linear(in_features=msg_dim,out_features=msg_dim)
            )
            self.tar_mlp=nn.Sequential(
                nn.Linear(in_features=mem_dim+mem_dim+time_dim,out_features=msg_dim),
                nn.ReLU(),
                nn.Linear(in_features=msg_dim,out_features=msg_dim)
            )

    def create_message(self,
            src,
            tar,
            event_t
        ):
        """
        Notice:
            - msg_fn="concat"인 경우 msg_dim은 input concat 차원과 같아야 한다.
        Input:
            src: [B,]
            tar: [B,]
            event_t: [B,]
        Output:
            src_msg: [B,msg_dim]
            tar_msg: [B,msg_dim]
        """
        src_mem=self.memory.get_batch_memory(batch_node=src) # [B,mem_dim]
        src_ts=self.memory.get_batch_timespan(batch_node=src,batch_t=event_t) # [B,1]
        src_ts_encoding=self.time_encoder(src_ts) # [B,time_dim]

        tar_mem=self.memory.get_batch_memory(batch_node=tar) # [B,mem_dim]
        tar_ts=self.memory.get_batch_timespan(batch_node=tar,batch_t=event_t) # [B,1]
        tar_ts_encoding=self.time_encoder(tar_ts) # [B,time_dim]

        src_msg=torch.concat(
            [
                src_mem,
                tar_mem,
                src_ts_encoding
            ],
            dim=-1
        ) # [B,mem_dim+mem_dim+time_dim]
        tar_msg=torch.concat(
            [
                tar_mem,
                src_mem,
                tar_ts_encoding,
            ],
            dim=-1
        ) # [B,mem_dim+mem_dim+time_dim]

        if self.msg_fn=="mlp":
            src_msg=self.src_mlp(src_msg) # [B,msg_dim]
            tar_msg=self.tar_mlp(tar_msg) # [B,msg_dim]
        return src_msg,tar_msg

    def aggregate_message(self,
            src,
            tar,
            src_msg,
            tar_msg,
            event_t
        ):
        """
        Memory Aggregation
        Input:
            src: [B,]
            tar: [B,]
            src_msg: [B,msg_dim]
            tar_msg: [B,msg_dim]
            event_t: [B,]
        Output:
            aggr_node: [unique_N,]
            aggr_msg: [unique_N,msg_dim]
        """
        nodes=torch.concat([src,tar],dim=0) # [2B,]
        msgs=torch.concat([src_msg,tar_msg],dim=0) # [2B,msg_dim]
        times=torch.concat([event_t,event_t],dim=0) # [2B,]

        # event time 순으로 오름차순 정렬
        sorted_idx=torch.argsort(times)
        nodes=nodes[sorted_idx]
        msgs=msgs[sorted_idx]
        times=times[sorted_idx]

        msg_dict={}
        for node,msg,t in zip(nodes,msgs,times):
            node=node.item()
            t=t.item()
            if node not in msg_dict:
                msg_dict[node]=[]
            msg_dict[node].append((msg,t))

        aggr_node=[]
        aggr_msg=[]
        match self.aggr_fn:
            case "last":
                aggr_node=[
                    node
                    for node in msg_dict.keys()
                ]
                aggr_msg=[
                    msg_dict[node][-1][0]
                    for node in msg_dict.keys()
                ]
            case "mean":
                aggr_node=[
                    node
                    for node in msg_dict.keys()
                ]
                aggr_msg=[
                    torch.mean(
                        torch.stack([msg for msg,_ in msg_dict[node]],dim=0),
                        dim=0
                    )
                    for node in msg_dict.keys()
                ]
        aggr_node=torch.tensor(aggr_node,device=src.device) # [unique_N,]
        aggr_msg=torch.stack(aggr_msg,dim=0) # [unique_N,msg_dim]
        return aggr_node,aggr_msg

    def update_memory_implement(self,aggr_node,aggr_msg):
        return NotImplemented

    def update_memory(self,src,tar,event_t):
        """
        자식 class의 update_memory 실행으로 자식 class에서 구현된 update_memory_implement 호출
        
        Input:
            src: [B,]
            tar: [B,]
            event_t: [B,]
        """
        src_msg,tar_msg=self.create_message(
            src=src,
            tar=tar,
            event_t=event_t
        )
        aggr_node,aggr_msg=self.aggregate_message(
            src=src,
            tar=tar,
            src_msg=src_msg,
            tar_msg=tar_msg,
            event_t=event_t
        )
        return self.update_memory_implement(
            aggr_node=aggr_node,
            aggr_msg=aggr_msg
        )

class GRUMemoryUpdater(MemoryUpdater):
    def __init__(self,
            mem_dim:int,
            msg_dim:int,
            time_dim:int,
            time_encoder:TimeEncoder,
            memory:Memory,
            msg_fn:Literal["concat","mlp"]="concat",
            aggr_fn:Literal["last","mean"]="last"
        ):
        super(GRUMemoryUpdater,self).__init__(
            mem_dim=mem_dim,
            msg_dim=msg_dim,
            time_dim=time_dim,
            time_encoder=time_encoder,
            memory=memory,
            msg_fn=msg_fn,
            aggr_fn=aggr_fn
        )
        self.memory_updater=nn.GRUCell(
            input_size=msg_dim,
            hidden_size=mem_dim
        )

    def update_memory_implement(self,aggr_node,aggr_msg):
        """
        Input:
            aggr_node: [unique_N,]
            aggr_msg: [unique_N,msg_dim]
        """
        pre_memory=self.memory.get_batch_memory(batch_node=aggr_node) # [unique_N,mem_dim]
        updated_memory=self.memory_updater(aggr_msg,pre_memory) # [unique_N,mem_dim]
        self.memory.update_memory_state(batch_node=aggr_node,batch_memory=updated_memory)

class RNNMemoryUpdater(MemoryUpdater):
    def __init__(self,
            mem_dim:int,
            msg_dim:int,
            time_dim:int,
            time_encoder:TimeEncoder,
            memory:Memory,
            msg_fn:Literal["concat","mlp"]="concat",
            aggr_fn:Literal["last","mean"]="last"
        ):
        super(GRUMemoryUpdater,self).__init__(
            mem_dim=mem_dim,
            msg_dim=msg_dim,
            time_dim=time_dim,
            time_encoder=time_encoder,
            memory=memory,
            msg_fn=msg_fn,
            aggr_fn=aggr_fn
        )
        self.memory_updater=nn.RNNCell(
            input_size=msg_dim,
            hidden_size=mem_dim
        )

    def update_memory_implement(self,aggr_node,aggr_msg):
        """
        Input:
            aggr_node: [unique_N,]
            aggr_msg: [unique_N,msg_dim]
        """
        pre_memory=self.memory.get_batch_memory(batch_node=aggr_node) # [unique_N,mem_dim]
        updated_memory=self.memory_updater(aggr_msg,pre_memory) # [unique_N,mem_dim]
        self.memory.update_memory_state(batch_node=aggr_node,batch_memory=updated_memory)