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
    def get_sample_loader(
            n_sample:int,
            n_pair:int,
            max_hop:int|None,
            data_loader:DataLoader,
            graph:TemporalGraph
        ):
        """
        Return:
            sample_loader
        """
        sample_loader=[]
        for _,_,event_t,_ in tqdm(data_loader,desc=f"Compute TR sample..."):
            query_time=event_t[-1].item()
            sample=graph.random_TR_sampling(
                n_sample=n_sample,
                n_pair=n_pair,
                query_time=query_time,
                max_hop=max_hop
            )
            sample_loader.append(sample)
        return sample_loader

    @staticmethod
    def check_sample_loader(sample_loader:list):
        pos_counts=[]
        neg_counts=[]
        for sample in sample_loader:
            label=sample["label"]
            pos_counts.append((label==1).sum().item())
            neg_counts.append((label==0).sum().item())
        avg_pos=sum(pos_counts)/len(pos_counts)
        avg_neg=sum(neg_counts)/len(neg_counts)
        print(
            f"Avg pos: {avg_pos:.2f}, "
            f"Avg neg: {avg_neg:.2f}, "
            f"Avg total: {avg_pos+avg_neg:.2f}"
        )

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
