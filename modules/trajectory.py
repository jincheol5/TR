import torch
import numpy as np

"""
To Do List:
- 0번 노드 dummy node 처리 -> 항상 이웃 노드 아무것도 없도록
"""
class Trajectory:
    """
    각 node의 intermediate TR result 저장

    TR: dict of node intermediate TR result
        key: node_id
        value: TR result, 1 or 0 
    """
    def __init__(self,
            source_id:int=1,
            device:torch.device=torch.device("cpu")
        ):
        self.TR={}
        self.source_id=source_id
        self.device=device

        # set source node TR result to 1
        self.TR[source_id]=1

        # init dummy node TR result to 0
        self.TR[0]=0
    
    def update_trajectory(self,batch_events:list):
        """
        Input:
            event: List of tuple (src,tar,timestamp)
        """
        for event in batch_events:
            src,tar,_=event
            if src not in self.TR:
                self.TR[src]=0
            if tar not in self.TR:
                self.TR[tar]=0
    
    def update_TR(self,tar,tar_TR):
        """
        Input:
            tar: [B,]
            tar_TR: [B,]
        """
        for node,result in zip(tar,tar_TR):
            self.TR[node.item()]=result.item()
    
    def get_batch_TR(self,batch_node):
        """
        Input:
            batch_node: [B,]
        Output:
            batch_TR: [B,1]
        """
        batch_TR=torch.tensor(
            [
                self.TR[node.item()]
                for node in batch_node
            ],
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(1)
        return batch_TR