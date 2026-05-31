import torch
import numpy as np

"""
To Do List:
- 0번 노드 dummy node 처리 -> 항상 이웃 노드 아무것도 없도록
"""
class TemporalGraph:
    """
    node_ft: dict of each node feature
        key: node_id
        value: feature tensor
        dummy node id: 0
        dummy node feature: zero tensor
    neighbor: dict of each node's in-neighbor id list
        key: node_id
        value: list of neighbor node id
        dummy node's neighbor: []
    neighbor_t: dict of each node's in-neighbor interact timestamp
        key: node_id
        value: list of neighbor interact timestamp
        dummy node's neighbor_t: []
    """
    def __init__(self,
            node_dim:int=32,
            device:torch.device=torch.device("cpu")
        ):
        self.node_ft={}
        self.neighbor={}
        self.neighbor_t={}
        self.node_dim=node_dim
        self.device=device

        # init dummy node feature
        self.node_ft[0]=torch.zeros(self.node_dim,device=self.device)
        self.neighbor[0]=[]
        self.neighbor_t[0]=[]

    def set_device(self,
            device:torch.device=torch.device("cpu")
        ):
        self.device=device
        for node in self.node_ft.keys():
            self.node_ft[node].to(self.device)

    def update_graph(self,batch_events:list):
        """
        Input:
            event: List of tuple (src,tar,timestamp)
        """
        for event in batch_events:
            src,tar,timestamp=event
            if src not in self.node_ft:
                self.node_ft[src]=torch.ones(self.node_dim,device=self.device)
            if tar not in self.node_ft:
                self.node_ft[tar]=torch.ones(self.node_dim,device=self.device)
            if tar not in self.neighbor:
                self.neighbor[tar]=[]
                self.neighbor_t[tar]=[]
            self.neighbor[tar].append(src)
            self.neighbor_t[tar].append(timestamp)

    def find_temporal_neighbor(self,tar,cut_time):
        """
        temporal neighbor 없을 경우 [],[] 반환
        """
        if tar not in self.neighbor:
            return [],[]
        t_np=np.array(self.neighbor_t[tar])
        idx=np.searchsorted(t_np,cut_time,side="left")
        return self.neighbor[tar][:idx],self.neighbor_t[tar][:idx]

    def get_data_for_embedding(self,batch_tar,batch_t):
        """
        Input:
            batch_tar: [B,]
            batch_t: [B,]
        Output
            batch_tar_ts: [B,]
            batch_n: [B,N]
            batch_n_t: [B,N]
            batch_n_ts: [B,N], 이웃 노드들과의 timespan
            batch_n_mask: [B,N]
        """
        temporal_n=[
            self.find_temporal_neighbor(
                tar=tar.item(),
                cut_time=timestamp.item()
            )
            for tar,timestamp in zip(batch_tar,batch_t)
        ] # List of tuple (neighbor=[...],neighbor_t=[...])
        n_list=[result[0] for result in temporal_n]
        n_t_list=[result[1] for result in temporal_n]
        max_n=max(len(n) for n in n_list)

        if max_n==0:
            """
            target node들의 이웃 노드들이 단 한개도 없는 경우 dummy node를 이웃으로 반환, mask는 False
            """
            batch_size=batch_tar.size(0)
            batch_tar_ts=torch.zeros((batch_size,),dtype=torch.float32,device=self.device) # [B,]
            batch_n=torch.zeros((batch_size,1),dtype=torch.long,device=self.device) # [B,1]
            batch_n_t=torch.zeros((batch_size,1),dtype=torch.float32,device=self.device) # [B,1]
            batch_n_ts=torch.zeros((batch_size,1),dtype=torch.float32,device=self.device) # [B,1]
            batch_n_mask=torch.zeros((batch_size,1),dtype=torch.bool,device=self.device) # [B,1]
        else:
            batch_size=batch_tar.size(0)
            batch_tar_ts=torch.zeros((batch_size,),dtype=torch.float32,device=self.device) # [B,]
            batch_n=torch.zeros((batch_size,max_n),dtype=torch.long,device=self.device) # [B,max_n]
            batch_n_t=torch.zeros((batch_size,max_n),dtype=torch.float32,device=self.device) # [B,max_n]
            batch_n_ts=torch.zeros((batch_size,max_n),dtype=torch.float32,device=self.device) # [B,max_n]
            batch_n_mask=torch.zeros((batch_size,max_n),dtype=torch.bool,device=self.device) # [B,max_n]

            for idx,(neighbors,timestamps) in enumerate(zip(n_list,n_t_list)):
                n_len=len(neighbors)
                if n_len==0:
                    continue
                neighbors_tensor=torch.tensor(neighbors,dtype=torch.long,device=self.device)
                timestamps_tensor=torch.tensor(timestamps,dtype=torch.float32,device=self.device)

                batch_n[idx,:n_len]=neighbors_tensor
                batch_n_t[idx,:n_len]=timestamps_tensor

                batch_n_ts[idx,:n_len]=torch.abs(
                    batch_t[idx]-timestamps_tensor # broadcasting
                )
                batch_n_mask[idx,:n_len]=True

        return {
            "batch_tar_ts": batch_tar_ts,
            "batch_n": batch_n,
            "batch_n_t": batch_n_t,
            "batch_n_ts": batch_n_ts,
            "batch_n_mask" : batch_n_mask
        }

    def get_batch_node_feature(self,batch_node):
        """
        Input:
            batch_node: [B,]
        Output:
            batch_ft: [B,node_dim]
        """
        batch_ft=torch.stack(
            [self.node_ft[node.item()] for node in batch_node],
            dim=0
        ) # [B,node_dim]
        return batch_ft

    def get_batch_neighbor_feature(self,batch_n,batch_n_mask):
        """
        Input:
            batch_n: [B,N]
            batch_n_mask: [B,N]
        Output:
            batch_n_ft: [B,N,node_dim]
        """
        padding_ft=torch.zeros(self.node_dim,device=self.device)
        batch_n_ft=torch.stack([
            torch.stack([
                self.node_ft[node_id.item()]
                if is_valid.item()
                else padding_ft
                for node_id,is_valid in zip(neighbors,masks)
            ],dim=0)
            for neighbors,masks in zip(batch_n,batch_n_mask)
        ],dim=0) # [B,N,node_dim]
        return batch_n_ft

