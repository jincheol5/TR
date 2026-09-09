import torch

class TR_Sampling:
    @staticmethod
    def random_TR_sampling(
            sources:list,
            n_pair:int,
            query_time:float,
            TR_label:torch.Tensor
        )->dict[str,torch.Tensor]:
        """
        각 source 마다 n_pair 개의 positive/negative dst node를 random sampling.
        sources는 중복이 없도록 한다.
        dst node는 src node 자기자신과 padding node (id=0)는 제외한다.
        각 source에 대해 sampling 된 dst node는 중복이 없도록 한다. 
        pair 수가 비균형 할 경우에도 부족한 대로 수를 유지한다. (ex: pos_pair:10, neg_pair:5)
        한 쪽 밖에 없는 경우에도 유효한 쪽만 유지하여 sampling 한다. (ex: pos_pair:0, neg_pair:10)
        source 별 sampling이 끝난 후, positive/negative 별로 sample 개수(n_pair x len(sources))가 부족한 경우 다음 과정 수행:
        - sources에 없던 source를 랜덤 선정
        - 부족한 positive/negative sample 수가 n_pair 이상인 경우, n_pair 수 만큼 random TR Sampling 수행. 
        - 부족한 positive/negative sample 수가 n_pair 미만인 경우, 부족한 수 만큼 random TR Sampling 수행.
        - 부족한 sample 수 채워질 때 까지 반복.

        Input:
            sources: sampling 할 source node list
            n_pair: source 당 sampling할 pos/neg pair node 개수
            query_time: float
            TR_label: [N+1,N+1] bool tensor, reachable하면 True, unreachable하면 False
        Return:
            Return: dict
                src: [n_sample,] long tensor
                dst: [n_sample,] long tensor
                label: [n_sample,] float tensor (1.0 or 0.0)
                query_t: [n_sample,] float tensor
                pos_mask: [n_sample,] bool tensor
        """
        n_node=TR_label.size(0)-1
        n_sample=n_pair*len(sources)

        sampled_src=[]
        sampled_dst=[]
        sampled_label=[]

        n_pos=0
        n_neg=0

        ### 기존 sources에서 sampling
        for source in sources:
            source=int(source)

            pos_nodes=torch.nonzero(TR_label[source],as_tuple=False).flatten()
            neg_nodes=torch.nonzero(~TR_label[source],as_tuple=False).flatten()

            # padding node, 자기 자신 제외
            pos_nodes=pos_nodes[
                (pos_nodes!=0)&(pos_nodes!=source)
            ]
            neg_nodes=neg_nodes[
                (neg_nodes!=0)&(neg_nodes!=source)
            ]

            # positive sampling
            n_pos_sample=min(n_pair,pos_nodes.numel())
            if n_pos_sample>0:
                selected=pos_nodes[
                    torch.randperm( # 0 ~ len(pos_nodes)-1의 index를 무작위 순서로 생성
                        pos_nodes.numel(),
                        device=pos_nodes.device
                    )[:n_pos_sample] 
                ]
                sampled_src.append(
                    torch.full_like(
                        selected,
                        source,
                        dtype=torch.long
                    )
                )
                sampled_dst.append(selected.long())
                sampled_label.append(
                    torch.ones_like(
                        selected,
                        dtype=torch.float32
                    )
                )
                n_pos+=n_pos_sample

            # negative sampling
            n_neg_sample=min(n_pair,neg_nodes.numel())
            if n_neg_sample>0:
                selected=neg_nodes[
                    torch.randperm(
                        neg_nodes.numel(),
                        device=neg_nodes.device
                    )[:n_neg_sample]
                ]
                sampled_src.append(
                    torch.full_like(
                        selected,
                        source,
                        dtype=torch.long
                    )
                )
                sampled_dst.append(selected.long())
                sampled_label.append(
                    torch.zeros_like(
                        selected,
                        dtype=torch.float32
                    )
                )
                n_neg+=n_neg_sample

        ### 기존 sources에 포함되지 않은 source 후보
        used_sources=set(map(int,sources))
        extra_sources=[
            source
            for source in range(1,n_node+1)
            if source not in used_sources
        ]
        if extra_sources:
            perm=torch.randperm(len(extra_sources)).tolist()
            extra_sources=[
                extra_sources[i]
                for i in perm
            ]

        ### 부족한 positive / negative sample 보충
        for source in extra_sources:
            pos_deficit=max(0,n_sample-n_pos)
            neg_deficit=max(0,n_sample-n_neg)
            if pos_deficit==0 and neg_deficit==0:
                break

            pos_nodes=torch.nonzero(
                TR_label[source],
                as_tuple=False
            ).flatten()

            neg_nodes=torch.nonzero(
                ~TR_label[source],
                as_tuple=False
            ).flatten()

            pos_nodes=pos_nodes[
                (pos_nodes!=0)&(pos_nodes!=source)
            ]
            neg_nodes=neg_nodes[
                (neg_nodes!=0)&(neg_nodes!=source)
            ]

            # positive 보충
            n_pos_sample=min(n_pair,pos_deficit,pos_nodes.numel())
            if n_pos_sample>0:
                selected=pos_nodes[
                    torch.randperm(
                        pos_nodes.numel(),
                        device=pos_nodes.device
                    )[:n_pos_sample]
                ]
                sampled_src.append(
                    torch.full_like(
                        selected,
                        source,
                        dtype=torch.long
                    )
                )
                sampled_dst.append(selected.long())
                sampled_label.append(
                    torch.ones_like(
                        selected,
                        dtype=torch.float32
                    )
                )
                n_pos+=n_pos_sample

            # negative 보충
            n_neg_sample=min(n_pair,neg_deficit,neg_nodes.numel())
            if n_neg_sample>0:
                selected=neg_nodes[
                    torch.randperm(
                        neg_nodes.numel(),
                        device=neg_nodes.device
                    )[:n_neg_sample]
                ]
                sampled_src.append(
                    torch.full_like(
                        selected,
                        source,
                        dtype=torch.long
                    )
                )
                sampled_dst.append(selected.long())
                sampled_label.append(
                    torch.zeros_like(
                        selected,
                        dtype=torch.float32
                    )
                )
                n_neg+=n_neg_sample

        ### 결과 tensor 생성
        src=torch.cat(sampled_src)
        dst=torch.cat(sampled_dst)
        label=torch.cat(sampled_label)
        query_t=torch.full_like(
            label,
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

    """
    << Hard TR Sampling >>
    positive_hard_TR_sampling
        → 한 source의 positive hard dst 선택

    negative_hard_TR_sampling
        → 한 source의 negative hard dst 선택

    hard_TR_sampling
        → 여러 source 관리
        → 부족량 계산
        → hard / random 선택
        → 최종 tensor 구성
    """
    @staticmethod
    def positive_hard_TR_sampling(
            source:int,
            n_pair:int,
            start_query_time:float,
            end_query_time:float,
            TR_label:torch.Tensor,
            TR_hop:torch.Tensor,
            TR_last_t:torch.Tensor,
            pos_hard_ratio:float=0.5
        )->torch.Tensor:
        """
        pos_hard_ratio: 2-hop 이상 TR 비율
        """
        pos_nodes=torch.nonzero(TR_label[source]).flatten()
        pos_nodes=pos_nodes[
            (pos_nodes!=0)&
            (pos_nodes!=source)
        ]

        n_multi_hop=int(n_pair*pos_hard_ratio)
        n_one_hop=n_pair-n_multi_hop

        one_hop_nodes=pos_nodes[
            TR_hop[source,pos_nodes]==1
        ]
        multi_hop_nodes=pos_nodes[
            TR_hop[source,pos_nodes]>=2
        ]

        # start_query_time 이후 reachable 된 node 우선
        one_hop_priority_mask=(
            (TR_last_t[source,one_hop_nodes]>start_query_time)&
            (TR_last_t[source,one_hop_nodes]<=end_query_time)
        )
        multi_hop_priority_mask=(
            (TR_last_t[source,multi_hop_nodes]>start_query_time)&
            (TR_last_t[source,multi_hop_nodes]<=end_query_time)
        )

        one_hop_priority=one_hop_nodes[one_hop_priority_mask]
        one_hop_other=one_hop_nodes[~one_hop_priority_mask]

        multi_hop_priority=multi_hop_nodes[multi_hop_priority_mask]
        multi_hop_other=multi_hop_nodes[~multi_hop_priority_mask]

        if one_hop_priority.numel()>0:
            one_hop_priority=one_hop_priority[
                torch.randperm(
                    one_hop_priority.numel(),
                    device=one_hop_priority.device
                )
            ]

        if one_hop_other.numel()>0:
            one_hop_other=one_hop_other[
                torch.randperm(
                    one_hop_other.numel(),
                    device=one_hop_other.device
                )
            ]

        if multi_hop_priority.numel()>0:
            multi_hop_priority=multi_hop_priority[
                torch.randperm(
                    multi_hop_priority.numel(),
                    device=multi_hop_priority.device
                )
            ]

        if multi_hop_other.numel()>0:
            multi_hop_other=multi_hop_other[
                torch.randperm(
                    multi_hop_other.numel(),
                    device=multi_hop_other.device
                )
            ]

        one_hop_nodes=torch.cat([one_hop_priority,one_hop_other])
        multi_hop_nodes=torch.cat([multi_hop_priority,multi_hop_other])

        # multi-hop을 지정 비율만큼 샘플링
        n_multi_hop_sample=min(
            n_multi_hop,
            multi_hop_nodes.numel()
        )
        selected_multi_hop=multi_hop_nodes[
            :n_multi_hop_sample
        ]

        # multi-hop 부족분을 1-hop으로 보충
        multi_hop_shortage=(
            n_multi_hop-n_multi_hop_sample
        )

        n_one_hop_sample=min(
            n_one_hop+multi_hop_shortage,
            one_hop_nodes.numel()
        )
        selected_one_hop=one_hop_nodes[
            :n_one_hop_sample
        ]

        return torch.cat([
            selected_one_hop,
            selected_multi_hop
        ])

    @staticmethod
    def negative_hard_TR_sampling(
            source:int,
            n_pair:int,
            SR_label:torch.Tensor,
            SR_hop:torch.Tensor,
            TR_label:torch.Tensor,
            neg_hard_ratio:float=0.5
        )->torch.Tensor:
        """
        neg_hard_ratio: SR이지만 not TR인 sample 비율
        """
        neg_nodes=torch.nonzero(~TR_label[source]).flatten()
        neg_nodes=neg_nodes[
            (neg_nodes!=0)&
            (neg_nodes!=source)
        ]

        n_SR=int(n_pair*neg_hard_ratio)
        n_both_unreachable=n_pair-n_SR

        # static / temporal 모두 unreachable
        both_unreachable_nodes=neg_nodes[~SR_label[source,neg_nodes]]

        # static에서는 2-hop 이상 reachable / temporal에서는 unreachable
        SR_nodes=neg_nodes[SR_label[source,neg_nodes]&(SR_hop[source,neg_nodes]>=2)]

        if both_unreachable_nodes.numel()>0:
            both_unreachable_nodes=both_unreachable_nodes[
                torch.randperm(
                    both_unreachable_nodes.numel(),
                    device=both_unreachable_nodes.device
                )
            ]

        if SR_nodes.numel()>0:
            SR_nodes=SR_nodes[
                torch.randperm(
                    SR_nodes.numel(),
                    device=SR_nodes.device
                )
            ]

        n_both_unreachable_sample=min(n_both_unreachable,both_unreachable_nodes.numel())
        n_SR_sample=min(n_SR,SR_nodes.numel())

        selected_both_unreachable=both_unreachable_nodes[:n_both_unreachable_sample]
        selected_SR=SR_nodes[:n_SR_sample]

        # 한 조건이 부족하면 다른 조건의 남은 후보로 보충
        deficit=(n_pair-n_both_unreachable_sample-n_SR_sample)

        remain=torch.cat([
            both_unreachable_nodes[n_both_unreachable_sample:],
            SR_nodes[n_SR_sample:]
        ])

        if remain.numel()>0:
            remain=remain[
                torch.randperm(
                    remain.numel(),
                    device=remain.device
                )
            ]

        return torch.cat([
            selected_both_unreachable,
            selected_SR,
            remain[:deficit]
        ])

    @staticmethod
    def hard_TR_sampling(
            sources:list,
            n_pair:int,
            start_query_time:float,
            end_query_time:float,
            SR_label:torch.Tensor,
            SR_hop:torch.Tensor,
            TR_label:torch.Tensor,
            TR_hop:torch.Tensor,
            TR_last_t:torch.Tensor,
            pos_hard_ratio:float=0.5,
            neg_hard_ratio:float=0.5
        )->dict[str,torch.Tensor]:
        """
        << Hard TR Sampling >>
        positive node pair sampling:
        - start_query_time 이전에는 unreachable이였으나, start_query_time 이후로 reachable 해진 dst node들을 우선 sampling.
        - n_pair의 20%는 1-hop에 도달 가능한 dst node, 80%는 2-hop 이상에서 도달 가능한 dst node로 sampling.
        - 2-hop 이상에서 도달 가능한 dst node가 부족한 경우는 1-hop에 도달 가능한 dst node로 채운다. 

        negative node pair sampling:
        - 20% of n_pair
            -> static path, temporal path 모두 unreachable한 dst node로 sampling. 
        - 80% of n_pair
            -> 2-hop 이상의 static path로는 reachable 하지만, temporal path로는 unreachable한 dst node로 sampling.  
        - 두 조건 모두 한 쪽이 부족한 경우 다른쪽으로 채운다.

        각 source 마다 n_pair 개의 positive/negative dst node hard TR sampling.
        sources는 중복이 없도록 한다.
        dst node는 src node 자기자신과 padding node (id=0)는 제외한다.
        각 source에 대해 sampling 된 dst node는 중복이 없도록 한다. 
        pair 수가 비균형 할 경우에도 부족한 대로 수를 유지한다. (ex: pos_pair:10, neg_pair:5)
        한 쪽 밖에 없는 경우에도 유효한 쪽만 유지하여 sampling 한다. (ex: pos_pair:0, neg_pair:10)
        source 별 sampling이 끝난 후, positive/negative 별로 sample 개수(n_pair x len(sources))가 부족한 경우 다음 과정 수행:
        - sources에 없던 source를 랜덤 선정
        - 부족한 positive/negative sample 수가 n_pair 이상인 경우, n_pair 수 만큼 Hard TR Sampling 조건 맞춰 수행. 
        - 부족한 positive/negative sample 수가 n_pair 미만인 경우, 부족한 수 만큼 random TR Sampling 수행.
        - 부족한 sample 수 채워질 때 까지 반복.

        Input:
            sources: sampling 할 source node list
            n_pair: source 당 sampling할 pos/neg pair node 개수
            query_time: float
            SR_label: [N+1,N+1] bool tensor, reachable하면 True, unreachable하면 False
            SR_hop: [N+1,N+1] int tensor, reachable 할 시 최소 hop 수 (unreachable하면 0)
            TR_label: [N+1,N+1] bool tensor, reachable하면 True, unreachable하면 False
            TR_hop: [N+1,N+1] int tensor, reachable 할 시 최소 hop 수 (unreachable하면 0)
            TR_last_t: [N+1,N+1] int tensor, reachable 할 시 update time (unreachable하면 0)
        Return:
            Return:
                src: [n_sample,] long tensor
                dst: [n_sample,] long tensor
                label: [n_sample,] float tensor (1.0 or 0.0)
                query_t: [n_sample,] float tensor, end_query_time으로 생성
                pos_mask: [n_sample,] bool tensor
        """
        n_node=TR_label.size(0)-1
        n_sample=n_pair*len(sources)

        sampled_src=[]
        sampled_dst=[]
        sampled_label=[]

        n_pos=0
        n_neg=0

        ### 기존 sources에서 Hard TR Sampling
        for source in sources:
            source=int(source)

            # positive
            pos_selected=TR_Sampling.positive_hard_TR_sampling(
                source=source,
                n_pair=n_pair,
                start_query_time=start_query_time,
                end_query_time=end_query_time,
                TR_label=TR_label,
                TR_hop=TR_hop,
                TR_last_t=TR_last_t,
                pos_hard_ratio=pos_hard_ratio
            )

            if pos_selected.numel()>0:
                sampled_src.append(
                    torch.full_like(
                        pos_selected,
                        source,
                        dtype=torch.long
                    )
                )
                sampled_dst.append(pos_selected)
                sampled_label.append(
                    torch.ones_like(
                        pos_selected,
                        dtype=torch.float32
                    )
                )
                n_pos+=pos_selected.numel()

            # negative
            neg_selected=TR_Sampling.negative_hard_TR_sampling(
                source=source,
                n_pair=n_pair,
                SR_label=SR_label,
                SR_hop=SR_hop,
                TR_label=TR_label,
                neg_hard_ratio=neg_hard_ratio
            )
            if neg_selected.numel()>0:
                sampled_src.append(
                    torch.full_like(
                        neg_selected,
                        source,
                        dtype=torch.long
                    )
                )
                sampled_dst.append(neg_selected)
                sampled_label.append(
                    torch.zeros_like(
                        neg_selected,
                        dtype=torch.float32
                    )
                )
                n_neg+=neg_selected.numel()

        ### 기존 sources에 포함되지 않은 source 후보
        used_sources=set(map(int,sources))
        extra_sources=[
            source
            for source in range(1,n_node+1)
            if source not in used_sources
        ]
        if extra_sources:
            perm=torch.randperm(
                len(extra_sources)
            ).tolist()
            extra_sources=[
                extra_sources[i]
                for i in perm
            ]

        ### 부족한 sample 보충
        for source in extra_sources:
            pos_deficit=max(0,n_sample-n_pos)
            neg_deficit=max(0,n_sample-n_neg)
            if pos_deficit==0 and neg_deficit==0:
                break

            ### Positive 보충
            if pos_deficit>0:
                n_pos_req=min(n_pair,pos_deficit)

                # deficit >= n_pair → Hard Sampling
                if pos_deficit>=n_pair:
                    selected=TR_Sampling.positive_hard_TR_sampling(
                        source=source,
                        n_pair=n_pair,
                        start_query_time=start_query_time,
                        end_query_time=end_query_time,
                        TR_label=TR_label,
                        TR_hop=TR_hop,
                        TR_last_t=TR_last_t,
                        pos_hard_ratio=pos_hard_ratio
                    )

                # deficit < n_pair → Random Sampling
                else:
                    pos_nodes=torch.nonzero(TR_label[source]).flatten()
                    pos_nodes=pos_nodes[
                        (pos_nodes!=0)&
                        (pos_nodes!=source)
                    ]
                    n_pos_sample=min(n_pos_req,pos_nodes.numel())
                    selected=pos_nodes[
                        torch.randperm(
                            pos_nodes.numel(),
                            device=pos_nodes.device
                        )[:n_pos_sample]
                    ]

                if selected.numel()>0:
                    sampled_src.append(torch.full_like(selected,source,dtype=torch.long))
                    sampled_dst.append(selected)
                    sampled_label.append(torch.ones_like(selected,dtype=torch.float32))
                    n_pos+=selected.numel()

            ### Negative 보충
            if neg_deficit>0:
                n_neg_req=min(n_pair,neg_deficit)

                # deficit >= n_pair → Hard Sampling
                if neg_deficit>=n_pair:
                    selected=TR_Sampling.negative_hard_TR_sampling(
                        source=source,
                        n_pair=n_pair,
                        SR_label=SR_label,
                        SR_hop=SR_hop,
                        TR_label=TR_label,
                        neg_hard_ratio=neg_hard_ratio
                    )

                # deficit < n_pair → Random Sampling
                else:
                    neg_nodes=torch.nonzero(~TR_label[source]).flatten()
                    neg_nodes=neg_nodes[
                        (neg_nodes!=0)&
                        (neg_nodes!=source)
                    ]
                    n_neg_sample=min(n_neg_req,neg_nodes.numel())
                    selected=neg_nodes[
                        torch.randperm(
                            neg_nodes.numel(),
                            device=neg_nodes.device
                        )[:n_neg_sample]
                    ]

                if selected.numel()>0:
                    sampled_src.append(torch.full_like(selected,source,dtype=torch.long))
                    sampled_dst.append(selected)
                    sampled_label.append(torch.zeros_like(selected,dtype=torch.float32))
                    n_neg+=selected.numel()

        ### 결과 tensor
        src=torch.cat(sampled_src)
        dst=torch.cat(sampled_dst)
        label=torch.cat(sampled_label)
        query_t=torch.full_like(
            label,
            float(end_query_time),
            dtype=torch.float32
        )
        return {
            "src":src,
            "dst":dst,
            "label":label,
            "query_t":query_t,
            "pos_mask":label.bool()
        }

