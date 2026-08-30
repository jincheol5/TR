import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from graph import TemporalGraph

class TemporalGraphDataset(Dataset):
    def __init__(self,df:pd.DataFrame):
        self.src=torch.tensor(df["u"].values,dtype=torch.long)
        self.dst=torch.tensor(df["i"].values,dtype=torch.long)
        self.t=torch.tensor(df["t"].values,dtype=torch.float32)
        self.edge=torch.tensor(df["idx"].values,dtype=torch.long)

    def __len__(self):
        return len(self.src)

    def __getitem__(self,idx):
        return self.src[idx],self.dst[idx],self.t[idx],self.edge[idx]

class TRSampleDataset(Dataset):
    """
    pos/neg pair 수가 다를 경우 작은 쪽의 수로 맞춰짐

    Input:
        sample: {
            "src": [N,] long tensor,
            "dst": [N,] long tensor,
            "label": [N,] tensor,
            "pos_mask": [N,] bool tensor,
            "pair_info": list
        }
    Return:
        각 item은 positive 1개, negative 1개로 구성
    """
    def __init__(self,
            sample:dict,
            query_time:float
        ):
        pos_mask=sample["pos_mask"]
        neg_mask=~pos_mask

        self.pos_src=sample["src"][pos_mask]
        self.pos_dst=sample["dst"][pos_mask]
        self.pos_label=sample["label"][pos_mask]
        self.pos_info=[
            info
            for info,is_pos in zip(
                sample["pair_info"],
                pos_mask.tolist()
            )
            if is_pos
        ]

        self.neg_src=sample["src"][neg_mask]
        self.neg_dst=sample["dst"][neg_mask]
        self.neg_label=sample["label"][neg_mask]
        self.neg_info=[
            info
            for info,is_pos in zip(
                sample["pair_info"],
                pos_mask.tolist()
            )
            if not is_pos
        ]

        self.n_sample=min(
            len(self.pos_src),
            len(self.neg_src)
        )
        self.pos_pair_t=torch.full(
            (self.n_sample,),
            query_time,
            dtype=torch.float
        )
        self.neg_pair_t=torch.full(
            (self.n_sample,),
            query_time,
            dtype=torch.float
        )

    def __len__(self):
        return self.n_sample

    def __getitem__(self,idx):
        return {
            "pos_src":self.pos_src[idx],
            "pos_dst":self.pos_dst[idx],
            "pos_label":self.pos_label[idx],
            "pos_info":self.pos_info[idx],
            "pos_pair_t":self.pos_pair_t[idx],
            "neg_src":self.neg_src[idx],
            "neg_dst":self.neg_dst[idx],
            "neg_label":self.neg_label[idx],
            "neg_info":self.neg_info[idx],
            "neg_pair_t":self.neg_pair_t[idx]
        }

    def getAll(self):
        return {
            "pos_src":self.pos_src,
            "pos_dst":self.pos_dst,
            "pos_label":self.pos_label,
            "pos_info":self.pos_info,
            "pos_pair_t":self.pos_pair_t,
            "neg_src":self.neg_src,
            "neg_dst":self.neg_dst,
            "neg_label":self.neg_label,
            "neg_info":self.neg_info,
            "neg_pair_t":self.neg_pair_t
        }

class TrainUtils:
    @staticmethod
    def split_graph_df(
            df:pd.DataFrame,
            train_ratio:float=0.7,
            val_ratio:float=0.15
        ):
        """
        Input:
            df
            train_ratio
            val_ratio
        Return:
            train_df
            val_df
            test_df
        """
        n=len(df)
        train_end=int(n*train_ratio)
        val_end=int(n*(train_ratio+val_ratio))
        train_df=df.iloc[:train_end].reset_index(drop=True)
        val_df=df.iloc[train_end:val_end].reset_index(drop=True)
        test_df=df.iloc[val_end:].reset_index(drop=True)
        return train_df,val_df,test_df

    @staticmethod
    def get_TR_result(
            graph:TemporalGraph,
            data_loader:DataLoader,
            max_hop:int
        )->dict[str,torch.Tensor]:
        """
        Input:
            graph
            data_loader
            max_hop
        Return:
            TR_result: dict
                TR_label: [seq_len,N+1,N+1], boolean tensor
                TR_hop: [seq_len,N+1,N+1], int8 tensor
                TR_last_t: [seq_len,N+1,N+1], int16 tensor
                TR_first_t: [seq_len,N+1,N+1], int16 tensor
        """
        n_node=graph.get_num_node()
        TR_result=torch.zeros(
            (len(data_loader),n_node+1,n_node+1),
            dtype=torch.bool
        )
        TR_hop=torch.zeros(
            (len(data_loader),n_node+1,n_node+1),
            dtype=torch.int8
        )
        TR_last_t=torch.zeros(
            (len(data_loader),n_node+1,n_node+1),
            dtype=torch.int16
        )
        TR_first_t=torch.zeros(
            (len(data_loader),n_node+1,n_node+1),
            dtype=torch.int16
        )
        for seq_idx,(_,_,event_t,_) in enumerate(tqdm(data_loader,desc="Compute TR result tensor...")):
            query_time=event_t.max().item()
            for source in range(1,n_node+1):
                TR_info=graph.compute_TR(
                    source=source,
                    query_time=query_time,
                    max_hop=max_hop
                )
                for dst in range(1,n_node+1):
                    info=TR_info[dst]
                    if not info["r"]:
                        continue

                    TR_result[seq_idx,source,dst]=True
                    TR_hop[seq_idx,source,dst]=int(info["hop"])
                    TR_first_t[seq_idx,source,dst]=int(info["first_t"])

                    # source node의 last_t는 -inf이므로 padding 값 0을 유지
                    if dst!=source:
                        TR_last_t[seq_idx,source,dst]=int(info["last_t"])
        return {
            "TR_label":TR_result,
            "TR_hop":TR_hop,
            "TR_last_t":TR_last_t,
            "TR_first_t":TR_first_t
        }

    # @staticmethod
    # def get_TR_sample_loader(
    #         graph:TemporalGraph,
    #         n_sample:int,
    #         n_pair:int,
    #         query_time:float,
    #         max_hop:int=5,
    #         batch_size:int=200
    #     ):
    #     """
    #     """
    #     TR_sample=graph.random_TR_sampling(
    #         n_sample=n_sample,
    #         n_pair=n_pair,
    #         query_time=query_time,
    #         max_hop=max_hop
    #     )
    #     print(f"Finish to generate TR sample!")

    #     # Dataset
    #     TR_sample_dataset=TRSampleDataset(sample=TR_sample,query_time=query_time)

    #     # DataLoader, query_time 기준으로 생성되었기 때문에 shuffle 가능
    #     TR_sample_loader=DataLoader(dataset=TR_sample_dataset,batch_size=batch_size,shuffle=True)
    #     return TR_sample_loader

    # @staticmethod
    # def get_TR_sample_dataset_list(
    #         graph:TemporalGraph,
    #         n_sample:int,
    #         n_pair:int,
    #         max_hop:int,
    #         data_loader:DataLoader
    #     ):
    #     """
    #     """
    #     TR_sample_dataset_list=[]
    #     for _,_,event_t,_ in tqdm(
    #             data_loader,
    #             desc="Generating TR samples for training..."
    #         ):
    #         query_time=event_t.max().item()
    #         TR_sample=graph.random_TR_sampling(
    #             n_sample=n_sample, 
    #             n_pair=n_pair,
    #             max_hop=max_hop,
    #             query_time=query_time
    #         )
    #         TR_sample_dataset=TRSampleDataset(
    #             sample=TR_sample,
    #             query_time=query_time
    #         )
    #         TR_sample_dataset_list.append(TR_sample_dataset)
    #     return TR_sample_dataset_list

class EarlyStopper:
    def __init__(self,
            patience:int=1
        ):
        self.patience=patience
        self.patience_count=0
        self.best_loss=np.inf
        self.best_state=None
        self.early_stop=False
    def __call__(self,
            val_loss:float,
            model:torch.nn.Module
        ):
        # val_loss가 NaN, Inf이면 즉시 early stop
        if not np.isfinite(val_loss): 
            print("Loss is NaN or Inf!")
            self.early_stop=True
            if self.best_state is not None:
                model.load_state_dict(self.best_state)
            return model

        # 첫 번째 validation에서는 비교할 이전 best가 없으므로 현재 loss와 모델을 그대로 best로 저장
        if self.best_state is None: 
            self.best_loss=val_loss
            self.best_state={
                key: value.detach().clone()
                for key,value in model.state_dict().items()
            }
            return model

        # val_loss가 개선 되지 않은 경우
        if self.best_loss<=val_loss: 
            self.patience_count+=1
            if self.patience<=self.patience_count:
                self.early_stop=True
                model.load_state_dict(self.best_state)
            return model

        # val_loss가 개선 된 경우
        self.patience_count=0
        self.best_loss=val_loss
        self.best_state={
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        return model
