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


class ReaCH_TGN_Utils:
    @staticmethod
    def Temporal_Augmentation(
            event_t:torch.Tensor,
            jitter_std:float=0.01,
            jitter_range:float=0.01,
        ):
        """
        ReaCH-TGN Temporal Augmentation.
        
        1. Event Drop -> NT-Xent Loss 계산 명확성을 위해 해당 과정 미구현
        2. Timestamp Jitter
            - Gaussian noise
            - Uniform jitter
        
        Timestamp 증강으로 batch 내 이벤트들의 순서 바뀌어도 그대로 유지

        Input:
            event_t: [B,]
            jitter_std: float
            jitter_range: float
        Return:
            augmented_event_t: [B,]
        """
        # Gaussian timestamp jitter
        augmented_event_t=event_t+torch.randn_like(event_t)*jitter_std
        # Uniform timestamp jitter
        augmented_event_t=augmented_event_t+(torch.rand_like(event_t)*2-1)*jitter_range
        return augmented_event_t

    @staticmethod
    def compute_NT_Xent_Loss(
            Z_src_A:torch.Tensor,
            Z_dst_A:torch.Tensor,
            Z_src_B:torch.Tensor,
            Z_dst_B:torch.Tensor,
            temperature:float=0.5
        ):
        """
        대조 학습을 위한 NT_Xent Loss 계산
        positive pair: 동일 노드의 다른 embedding 표현 = 동일 위치의 embedding

        Z_src_A: A augmentation view의 src node embedding 
        Z_dst_A: A augmentation view의 dst node embedding 
        Z_src_B: B augmentation view의 src node embedding 
        Z_dst_B: B augmentation view의 dst node embedding 

        Input:
            Z_src_A: [B,embed_dim]
            Z_dst_A: [B,embed_dim]
            Z_src_B: [B,embed_dim]
            Z_dst_B: [B,embed_dim]
            temperature: float
        Return:
            NT_Xent Loss
        """
        # 각 view의 node embedding 결합
        Z_A=torch.cat([Z_src_A,Z_dst_A],dim=0) # [2B,D]
        Z_B=torch.cat([Z_src_B,Z_dst_B],dim=0) # [2B,D]

        # cosine similarity 계산을 위한 normalization
        Z_A=F.normalize(Z_A,dim=1)
        Z_B=F.normalize(Z_B,dim=1)

        # 두 view 결합
        Z=torch.cat([Z_A,Z_B],dim=0) # [4B,D]
        n=Z_A.size(0) # 2B

        # pairwise cosine similarity
        logits=Z@Z.T/temperature # [4B,4B]

        # 자기 자신과의 similarity 제외
        mask=torch.eye(
            2*n,
            dtype=torch.bool,
            device=Z.device
        )
        logits=logits.masked_fill(mask,float("-inf"))

        # Positive pair index
        # Z_A[i] <-> Z_B[i]
        target=torch.cat([
            torch.arange(n,2*n,device=Z.device),
            torch.arange(n,device=Z.device)
        ]) # [4B]
        return F.cross_entropy(logits,target)

    def compute_hop_based_penalty(self,
            pair_hop:torch.Tensor,
            max_hop:int=5,
            gamma:float=1.0
        ):
        """
        Hop-Based Penalty
        Positive node pair loss에 대해서만 적용
        Hop이 짧을수록 더 큰 weight를 부여
        w_hop(h)=(H_max-h+1)^gamma

        Input:
            pair_hop: [N_pos,] long tensor
            max_hop: 최대 hop
            gamma: hop weight 감소 정도
        Return:
            weight: [N_pos,] float tensor
        """
        weight=(max_hop-pair_hop+1).float().pow(gamma)
        return weight # positive pair의 BCE loss를 sample별로 계산한 뒤 hop weight를 곱해주면 된다: pos_loss=pos_loss*hop_weight

    def compute_time_gap_penalty(self,
            pair_first_t:torch.Tensor,
            query_time:float,
            decay_lambda:float=1e-5
        ):
        """
        Time-Gap Penalty
        Positive node pair loss에 대해서만 적용
        Query time과 temporal path의 최초 event 시간 차이가 클수록 작은 weight를 부여

        decay_lambda = 시간 차이(time gap)가 커질 때 weight를 얼마나 빠르게 감소시킬지를 결정하는 지수 감쇠 계수
        Timestamp가 unix seconde 인 경우 decay_lambda=1e-5 정도 수준으로 해야함
        ReaCH-TGN의 경우 timestamp가 일 단위로 이산화 되었기 때문에 decay_lambda=0.01  

        w_time(Δt)=exp(-lambda*Δt)
        Δt=query_time-first_t

        Input:
            pair_first_t: [N_pos,] float tensor
            query_time: query 시점
            decay_lambda: 시간 감쇠 계수

        Return:
            weight: [N_pos,] float tensor
        """
        time_gap=query_time-pair_first_t
        weight=torch.exp(
            -decay_lambda*time_gap
        )
        return weight # 이후 hop penalty와 함께 사용 -> weight=hop_weight*time_weight


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
