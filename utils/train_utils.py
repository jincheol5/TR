import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from graph import TemporalGraph
from .TR_sampling import TR_Sampling

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
    def get_SR_result(
            graph:TemporalGraph,
            data_loader:DataLoader,
            max_hop:int
        ):
        """
        Static Reachability라 하더라도 계산에 사용되는 edge 정보는 query time 이전 edge들로 한다.

        Input:
            graph
            data_loader
            max_hop
        Return:
            SR_result: dict
                SR_label: [seq_len,N+1,N+1], boolean tensor
                SR_hop: [seq_len,N+1,N+1], int8 tensor
        """
        n_node=graph.get_num_node()
        SR_result=torch.zeros(
            (len(data_loader),n_node+1,n_node+1),
            dtype=torch.bool
        )
        SR_hop=torch.zeros(
            (len(data_loader),n_node+1,n_node+1),
            dtype=torch.int8
        )
        for seq_idx,(_,_,event_t,_) in enumerate(tqdm(data_loader,desc="Compute SR result tensor...")):
            query_time=event_t.max().item()
            for source in range(1,n_node+1):
                SR_info=graph.compute_SR(
                    source=source,
                    query_time=query_time,
                    max_hop=max_hop
                )
                for dst in range(1,n_node+1):
                    info=SR_info[dst]
                    if not info["r"]:
                        continue
                    SR_result[seq_idx,source,dst]=True
                    SR_hop[seq_idx,source,dst]=int(info["hop"])
        return {
            "SR_label":SR_result,
            "SR_hop":SR_hop
        }

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

    @staticmethod
    def get_coarse_grained_TR_sample_list(
            n_node:int,
            n_sample:int,
            n_pair:int,
            query_time:float,
            batch_size:int,
            TR_label:torch.Tensor
        )->list[dict[str,torch.Tensor]]:
        """
        Walk-based Model 학습용 TR_sample_list 반환.

        Input:
            n_node
            n_sample
            n_pair
            query_time
            batch_size
            TR_label: [batch_len,N+1,N+1] bool tensor
        Return:
            batch_TR_sample_list
        """
        if batch_size<=0:
            raise ValueError("batch_size must be positive.")

        TR_sample=TrainUtils.random_src_TR_sampling(
            n_node=n_node,
            n_sample=n_sample,
            n_pair=n_pair,
            query_time=query_time,
            TR_label=TR_label[-1]
        )

        n_TR_sample=TR_sample["src"].size(0)
        batch_TR_sample_list=[]
        for start_idx in range(0,n_TR_sample,batch_size):
            end_idx=min(start_idx+batch_size,n_TR_sample)
            batch_TR_sample={
                key:value[start_idx:end_idx]
                for key,value in TR_sample.items()
            }
            batch_TR_sample_list.append(batch_TR_sample)
        return batch_TR_sample_list

    @staticmethod
    def get_fine_grained_TR_sample_list(
            n_pair:int,
            data_loader:DataLoader,
            TR_label:torch.Tensor
        ):
        """
        GNN-based Model 학습용 TR_sample_list 반환.

        Input:
            n_pair
            data_loader
            TR_label: [batch_len,N+1,N+1] bool tensor
        Return:
            TR_sample_list
        """
        TR_sample_list=[]
        for batch_idx,(src,dst,event_t,_) in tqdm(
                enumerate(data_loader),
                desc="Generating TR samples..."
            ):
            sources=torch.unique(torch.cat([src,dst])).tolist()
            query_time=event_t.max().item()
            TR_sample=TrainUtils.random_TR_sampling(
                sources=sources,
                n_pair=n_pair,
                query_time=query_time,
                TR_label=TR_label[batch_idx]
            )
            TR_sample_list.append(TR_sample)
        return TR_sample_list




