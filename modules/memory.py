import torch
import numpy as np

"""
To Do List:
- 0번 노드 dummy node 처리 -> 항상 이웃 노드 아무것도 없도록
"""
class Memory:
    """
    memory: dict of node memory state
        key: node_id
        value: memory state vector tensor, init zero-vector
        dummy node id: 0
        dummy node feature: zero-vector
    interact_t: dict of node's interact time list
        key: node_id
        value: interact time list 
        dummy node id: 0
        dummy node value: []
    """
    def __init__(self,
            mem_dim:int=32,
            device:torch.device=torch.device("cpu")
        ):
        self.memory={}
        self.interact_t={}
        self.mem_dim=mem_dim
        self.device=device

        # init dummy node
        self.memory[0]=torch.zeros(self.mem_dim,device=self.device)
        self.interact_t[0]=[]

    def set_device(self,
            device:torch.device=torch.device("cpu")
        ):
        self.device=device
        for node in self.memory.keys():
            self.memory[node].to(self.device)

    def update_memory(self,batch_events:list):
        """
        Input:
            event: List of tuple (src,tar,timestamp)
        """
        for event in batch_events:
            src,tar,timestamp=event
            if src not in self.memory:
                self.memory[src]=torch.zeros(self.mem_dim,device=self.device)
                self.interact_t[src]=[]
            if tar not in self.memory:
                self.memory[tar]=torch.zeros(self.mem_dim,device=self.device)
                self.interact_t[tar]=[]
            self.interact_t[src].append(timestamp)
            self.interact_t[tar].append(timestamp)

    def update_memory_state(self,batch_node,batch_memory):
        """
        Input:
            batch_node: [B,]
            batch_memory: [B,mem_dim]
        """
        for node,memory_state in zip(batch_node,batch_memory):
            self.memory[node.item()]=memory_state

    def find_pre_interact_t(self,node,cut_time):
        """
        """
        if node not in self.memory:
            return 0
        t_np=np.array(self.interact_t[node])
        idx=np.searchsorted(t_np,cut_time,side="left")
        if idx==0:
            pre_interact_t=0
        else:
            pre_interact_t=self.interact_t[node][idx-1]
        return pre_interact_t

    def get_batch_memory(self,batch_node):
        """
        memory store에 없는 노드의 경우 zero_tensor 

        Input:
            batch_tar: [B,]
        Output:
            batch_memory: [B,mem_dim]
        """
        batch_memory=torch.stack(
            [
                self.memory[node.item()].detach()
                if node.item() in self.memory
                else torch.zeros(self.mem_dim,device=self.device)
                for node in batch_node
            ],
            dim=0
        ) # [B,mem_dim]
        return batch_memory

    def get_batch_pre_t(self,batch_node,batch_t):
        """
        Input:
            batch_node: [B,]
            batch_t: [B,]
        Output:
            batch_pre_t: [B,1]
        """
        pre_interact_t_list=[
            [
                self.find_pre_interact_t(
                    node=node.item(),
                    cut_time=timestamp.item()
                )
            ]
            for node,timestamp in zip(batch_node,batch_t)
        ]
        batch_pre_t=torch.tensor(pre_interact_t_list,device=self.device) # [B,1]
        return batch_pre_t # [B,1]

    def get_batch_timespan(self,batch_node,batch_t):
        """
        Input:
            batch_node: [B,]
            batch_t: [B,]
        Output:
            batch_ts: [B,1]
        """
        batch_pre_t=self.get_batch_pre_t(
            batch_node=batch_node,
            batch_t=batch_t
        )
        batch_t=batch_t.unsqueeze(-1) # -> [B,1]
        batch_ts=torch.abs(
            batch_t-batch_pre_t
        )
        return batch_ts
