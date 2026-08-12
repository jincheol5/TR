import pandas as pd
import torch
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
    def get_TR_sample_loader(
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
        for _,_,event_t,_ in data_loader:
            query_time=event_t[-1].item()
            sample=graph.random_TR_sampling(
                n_sample=n_sample,
                n_pair=n_pair,
                query_time=query_time,
                max_hop=max_hop
            )
            sample_loader.append({
                "src":sample["src"],
                "dst":sample["dst"],
                "label":sample["label"],
                "pos_mask":sample["pos_mask"],
                "pair_info":sample["pair_info"] # info: r, hop, first_t, last_t
            })
        return sample_loader