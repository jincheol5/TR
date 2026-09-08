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
            TR_last_t:torch.Tensor
        )->torch.Tensor:
        pos_nodes=torch.nonzero(TR_label[source]).flatten()
        pos_nodes=pos_nodes[
            (pos_nodes!=0)&
            (pos_nodes!=source)
        ]

        n_one=int(n_pair*0.2)
        n_two=n_pair-n_one

        one_nodes=pos_nodes[TR_hop[source,pos_nodes]==1]
        two_nodes=pos_nodes[TR_hop[source,pos_nodes]>=2]

        # start_query_time 이후 reachable 된 node 우선
        one_priority_mask=(
            (TR_last_t[source,one_nodes]>start_query_time)&
            (TR_last_t[source,one_nodes]<=end_query_time)
        )
        two_priority_mask=(
            (TR_last_t[source,two_nodes]>start_query_time)&
            (TR_last_t[source,two_nodes]<=end_query_time)
        )

        one_priority=one_nodes[one_priority_mask]
        one_other=one_nodes[~one_priority_mask]
        two_priority=two_nodes[two_priority_mask]
        two_other=two_nodes[~two_priority_mask]

        if one_priority.numel()>0:
            one_priority=one_priority[
                torch.randperm(
                    one_priority.numel(),
                    device=one_priority.device
                )
            ]
        if one_other.numel()>0:
            one_other=one_other[
                torch.randperm(
                    one_other.numel(),
                    device=one_other.device
                )
            ]
        if two_priority.numel()>0:
            two_priority=two_priority[
                torch.randperm(
                    two_priority.numel(),
                    device=two_priority.device
                )
            ]
        if two_other.numel()>0:
            two_other=two_other[
                torch.randperm(
                    two_other.numel(),
                    device=two_other.device
                )
            ]

        one_nodes=torch.cat([one_priority,one_other])
        two_nodes=torch.cat([two_priority,two_other])

        # 2-hop 이상 우선 80%
        n_two_sample=min(n_two,two_nodes.numel())
        selected_two=two_nodes[:n_two_sample]

        # 1-hop 20% + 2-hop 부족분
        n_one_sample=min(n_pair-n_two_sample,one_nodes.numel())
        selected_one=one_nodes[:n_one_sample]

        return torch.cat([
            selected_one,
            selected_two
        ])

    @staticmethod
    def negative_hard_TR_sampling(
            source:int,
            n_pair:int,
            SR_label:torch.Tensor,
            SR_hop:torch.Tensor,
            TR_label:torch.Tensor
        )->torch.Tensor:
        neg_nodes=torch.nonzero(~TR_label[source]).flatten()
        neg_nodes=neg_nodes[
            (neg_nodes!=0)&
            (neg_nodes!=source)
        ]

        n_both=int(n_pair*0.2)
        n_static=n_pair-n_both

        # static / temporal 모두 unreachable
        both_nodes=neg_nodes[~SR_label[source,neg_nodes]]

        # static 2-hop 이상 reachable / temporal unreachable
        static_nodes=neg_nodes[
            SR_label[source,neg_nodes]&
            (SR_hop[source,neg_nodes]>=2)
        ]

        if both_nodes.numel()>0:
            both_nodes=both_nodes[
                torch.randperm(
                    both_nodes.numel(),
                    device=both_nodes.device
                )
            ]
        if static_nodes.numel()>0:
            static_nodes=static_nodes[
                torch.randperm(
                    static_nodes.numel(),
                    device=static_nodes.device
                )
            ]

        n_both_sample=min(n_both,both_nodes.numel())
        n_static_sample=min(n_static,static_nodes.numel())
        selected_both=both_nodes[:n_both_sample]
        selected_static=static_nodes[:n_static_sample]

        # 한 조건이 부족하면 다른 조건의 남은 후보로 보충
        deficit=(n_pair-n_both_sample-n_static_sample)
        remain=torch.cat([
            both_nodes[n_both_sample:],
            static_nodes[n_static_sample:]
        ])

        if remain.numel()>0:
            remain=remain[
                torch.randperm(
                    remain.numel(),
                    device=remain.device
                )
            ]
        return torch.cat([
            selected_both,
            selected_static,
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
            TR_last_t:torch.Tensor
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
                TR_last_t=TR_last_t
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
                TR_label=TR_label
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
            perm=torch.randperm(len(extra_sources)).tolist()
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
                        TR_last_t=TR_last_t
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
                    sampled_src.append(
                        torch.full_like(
                            selected,
                            source,
                            dtype=torch.long
                        )
                    )
                    sampled_dst.append(selected)
                    sampled_label.append(
                        torch.ones_like(
                            selected,
                            dtype=torch.float32
                        )
                    )
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
                        TR_label=TR_label
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
                    sampled_src.append(
                        torch.full_like(
                            selected,
                            source,
                            dtype=torch.long
                        )
                    )
                    sampled_dst.append(selected)
                    sampled_label.append(
                        torch.zeros_like(
                            selected,
                            dtype=torch.float32
                        )
                    )
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

