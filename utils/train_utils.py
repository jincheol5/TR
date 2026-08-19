import pandas as pd
import numpy as np
import torch
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

class TRDataset(Dataset):
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

class TrainUtils:
    @staticmethod
    def split_graph_df(
            df:pd.DataFrame,
            train_ratio:float=0.7,
            val_ratio:float=0.1
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
    def get_sample_list(
            graph:TemporalGraph,
            max_hop:int,
            n_batch_sample:int,
            n_pair:int,
            data_loader:DataLoader
        ):
        """
        각 batch의 가장 큰 event_time을 query_time으로 하여 batch마다 graph.random_TR_sampling() 함수 수행 후 순서대로 리스트에 넣어서 리스트 반환하는 함수.
        data_loader는 TemporalGraphDataset에 대한 batch loader이다.
        각 sample 개수 = batch_size
        """
        sample_list=[]
        for _,_,event_t,_ in tqdm(
                data_loader,
                desc="Generating TR samples for training..."
            ):
            query_time=event_t.max().item()
            sample=graph.random_TR_sampling(
                n_sample=n_batch_sample, 
                n_pair=n_pair,
                max_hop=max_hop,
                query_time=query_time
            )
            sample_list.append(sample)
        return sample_list

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
