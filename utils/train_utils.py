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
    def random_src_TR_sampling(
            n_node:int,
            n_sample:int,
            n_pair:int,
            query_time:float,
            TR_label:torch.Tensor
        )->dict[str,torch.Tensor]:
        """
        pos_pair + neg_pair 개수가 n_sample 개가 될 때까지 sampling 반복합니다.
        매 반복마다 무작위로 source 추출 후 Reachable(positive) / Unreachable(negative) pair를 같은 수(n_pair)로 무작위 추출합니다.
        source node는 padding node를 제외하며, source node 간에 중복이 없도록 합니다.
        dst node에는 자기자신과 padding node를 제외합니다.
        pair 수가 비균형 할 경우 부족한 수만큼만 유지합니다. (ex: pos_pair:10, neg_pair:5 의 경우 pos_pair:5, neg_pair:5)
        한 쪽 밖에 없는 경우에도 유효한 쪽만 유지하여 sampling 합니다. (ex: pos_pair:0, neg_pair:10)

        Input:
            n_node
            n_sample: 총 pos/neg pair 개수 (짝수만 허용)
            n_pair: source 당 sampling할 pos/neg pair node 개수
            query_time:
            TR_label: [N+1,N+1] bool tensor
        Return:
            src: [n_sample,] long tensor
            dst: [n_sample,] long tensor
            label: [n_sample,] float tensor (1.0 or 0.0)
            query_t: [n_sample,] float tensor
            pos_mask: [n_sample,] bool tensor
        """
        if n_sample%2!=0:
            raise ValueError("n_sample must be even.")

        sampled_src=[]
        sampled_dst=[]
        sampled_label=[]
        n_sampled=0
        # padding node 0을 제외한 source를 중복 없이 무작위 순회
        sources=torch.randperm(n_node)+1
        for source in sources:
            if n_sampled>=n_sample:
                break

            candidate_mask=torch.ones(n_node+1,dtype=torch.bool)
            candidate_mask[0]=False
            candidate_mask[source]=False

            reachable=TR_label[source].bool()
            pos_nodes=torch.nonzero(candidate_mask&reachable).flatten()
            neg_nodes=torch.nonzero(candidate_mask&~reachable).flatten()
            remaining=n_sample-n_sampled

            if pos_nodes.numel()>0 and neg_nodes.numel()>0:
                # 양쪽 후보가 있으면 더 적은 쪽에 맞춰 균형 있게 sampling
                n_balanced=min(
                    n_pair,
                    pos_nodes.numel(),
                    neg_nodes.numel(),
                    remaining//2
                )
                n_pos_sample=n_balanced
                n_neg_sample=n_balanced
            elif pos_nodes.numel()>0:
                n_pos_sample=min(n_pair,pos_nodes.numel(),remaining)
                n_neg_sample=0
            elif neg_nodes.numel()>0:
                n_pos_sample=0
                n_neg_sample=min(n_pair,neg_nodes.numel(),remaining)
            else:
                continue

            if n_pos_sample>0:
                selected_pos=pos_nodes[
                    torch.randperm(pos_nodes.numel())[:n_pos_sample]
                ]
                sampled_src.append(
                    torch.full((n_pos_sample,),int(source),dtype=torch.long)
                )
                sampled_dst.append(selected_pos)
                sampled_label.append(
                    torch.ones(n_pos_sample,dtype=torch.float32)
                )

            if n_neg_sample>0:
                selected_neg=neg_nodes[
                    torch.randperm(neg_nodes.numel())[:n_neg_sample]
                ]
                sampled_src.append(
                    torch.full((n_neg_sample,),int(source),dtype=torch.long)
                )
                sampled_dst.append(selected_neg)
                sampled_label.append(
                    torch.zeros(n_neg_sample,dtype=torch.float32)
                )

            n_sampled+=n_pos_sample+n_neg_sample

        if len(sampled_src)>0:
            src=torch.cat(sampled_src)
            dst=torch.cat(sampled_dst)
            label=torch.cat(sampled_label)
        else:
            src=torch.empty(0,dtype=torch.long)
            dst=torch.empty(0,dtype=torch.long)
            label=torch.empty(0,dtype=torch.float32)

        query_t=torch.full(
            (src.numel(),),
            float(query_time),
            dtype=torch.float32
        )
        return {
            "src":src,
            "dst":dst,
            "label":label,
            "query_t":query_t,
            "pos_mask":label.bool()
        }

    @staticmethod
    def random_TR_sampling(
            sources:list,
            n_pair:int,
            query_time:float,
            TR_label:torch.Tensor
        )->dict[str,torch.Tensor]:
        """
        source 마다 Reachable(positive) / Unreachable(negative) pair를 같은 수(n_pair)로 무작위 추출합니다.
        source node는 중복이 없도록 합니다.
        dst node에는 자기자신과 padding node를 제외합니다.
        pair 수가 비균형 할 경우에도 부족한 대로 수를 유지합니다. (ex: pos_pair:10, neg_pair:5)
        한 쪽 밖에 없는 경우에도 유효한 쪽만 유지하여 sampling 합니다. (ex: pos_pair:0, neg_pair:10)

        Input:
            sources: sampling 할 source node list
            n_pair: source 당 sampling할 pos/neg pair node 개수
            query_time:
            TR_label: [N+1,N+1] bool tensor
        Return:
            src: [n_sample,] long tensor
            dst: [n_sample,] long tensor
            label: [n_sample,] float tensor (1.0 or 0.0)
            query_t: [n_sample,] float tensor
            pos_mask: [n_sample,] bool tensor
        """
        n_node=TR_label.size(0)-1
        sampled_src=[]
        sampled_dst=[]
        sampled_label=[]
        for source in sources:
            # destination 후보 설정
            candidate_mask=torch.ones(n_node+1,dtype=torch.bool)
            candidate_mask[0]=False
            candidate_mask[source]=False

            # positive/negative 후보 분리
            reachable=TR_label[source].bool()
            pos_mask=candidate_mask&reachable
            neg_mask=candidate_mask&~reachable
            pos_nodes=torch.nonzero(pos_mask).flatten()
            neg_nodes=torch.nonzero(neg_mask).flatten()

            # positive sampling
            n_pos_sample=min(n_pair,pos_nodes.numel())
            if n_pos_sample>0:
                pos_perm=torch.randperm(pos_nodes.numel())
                selected_pos=pos_nodes[pos_perm[:n_pos_sample]]
                sampled_src.append(
                    torch.full((n_pos_sample,),int(source),dtype=torch.long)
                )
                sampled_dst.append(selected_pos)
                sampled_label.append(
                    torch.ones(n_pos_sample,dtype=torch.float32)
                )

            # negative sampling
            n_neg_sample=min(n_pair,neg_nodes.numel())
            if n_neg_sample>0:
                neg_perm=torch.randperm(neg_nodes.numel())
                selected_neg=neg_nodes[neg_perm[:n_neg_sample]]
                sampled_src.append(
                    torch.full((n_neg_sample,),int(source),dtype=torch.long)
                )
                sampled_dst.append(selected_neg)
                sampled_label.append(
                    torch.zeros(n_neg_sample,dtype=torch.float32)
                )

        # sampled pair 결합
        if len(sampled_src)>0:
            src=torch.cat(sampled_src)
            dst=torch.cat(sampled_dst)
            label=torch.cat(sampled_label)
        else:
            src=torch.empty(0,dtype=torch.long)
            dst=torch.empty(0,dtype=torch.long)
            label=torch.empty(0,dtype=torch.float32)
        query_t=torch.full(
            (src.numel(),),
            float(query_time),
            dtype=torch.float32
        )
        pos_mask=label.bool()
        return {
            "src":src,
            "dst":dst,
            "label":label,
            "query_t":query_t,
            "pos_mask":pos_mask
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
