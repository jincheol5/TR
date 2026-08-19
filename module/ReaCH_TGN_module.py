import torch
import torch.nn.functional as F

class ReaCH_TGN_Module:
    def __init__(self):
        pass

    def Temporal_Augmentation(self,
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
        Return:
            augmented_event_t: [B,]
        """
        # Gaussian timestamp jitter
        augmented_event_t=event_t+torch.randn_like(event_t)*jitter_std
        # Uniform timestamp jitter
        augmented_event_t=augmented_event_t+(torch.rand_like(event_t)*2-1)*jitter_range
        return augmented_event_t

    def compute_NT_Xent_Loss(self,
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