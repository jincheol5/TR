import torch

class GraphAnalysis:

    @staticmethod
    def check_reach_ratio(
            reach_label:torch.Tensor
        ):  
        """
        all node pair에서 reachable한 node pair의 비율 계산.

        Input:
            reach_label: [N+1,N+1] boolean tensor
        """
        return reach_label.float().mean().item()

    @staticmethod
    def check_hard_positive_ratio(
            TR_label:torch.Tensor,
            TR_hop:torch.Tensor
        ):
        """
        all positive node pair 중 2-hop 이상에서 TR인 node pair 비율 확인
        self-loop, padding node 제외

        Input:
            TR_label: [N+1,N+1] boolean tensor
            TR_hop: [N+1,N+1] int tensor
        """
        TR_label=TR_label[1:,1:]
        TR_hop=TR_hop[1:,1:]
        n_node=TR_label.size(0)
        non_self_mask=~torch.eye(
            n_node,
            dtype=torch.bool,
            device=TR_label.device
        )
        pos_mask=TR_label&non_self_mask
        pos_hop=TR_hop[pos_mask]
        multi_hop_mask=pos_hop>=2
        n_total=pos_hop.numel()
        n_multi_hop=multi_hop_mask.sum().item()
        if n_total==0:
            return 0.0
        return n_multi_hop/n_total

    @staticmethod
    def check_hard_negative_ratio(
            TR_label:torch.Tensor,
            SR_label:torch.Tensor,
            SR_hop:torch.Tensor
        ):
        """
        all negative node pair 중 2-hop 이상의 static path로는 reachable 하지만, temporal path로는 unreachable한 node pair 비율 확인 
        padding node 제외 (self-loop는 자동으로 negative 제외)
        
        Input:
            TR_label: [N+1,N+1] boolean tensor
            SR_label: [N+1,N+1] boolean tensor
            SR_hop: [N+1,N+1] int tensor
        """
        TR_label=TR_label[1:,1:]
        SR_label=SR_label[1:,1:]
        SR_hop=SR_hop[1:,1:]
        neg_mask=~TR_label
        hard_neg_mask=(
            neg_mask &
            SR_label &
            (SR_hop >= 2)
        )
        n_total=neg_mask.sum().item()
        n_hard_neg=hard_neg_mask.sum().item()
        if n_total==0:
            return 0.0
        return n_hard_neg/n_total

    @staticmethod
    def check_hard_positive_ratio_in_sample(
            pos_src:torch.Tensor,
            pos_dst:torch.Tensor,
            TR_hop:torch.Tensor
        ):
        """
        positive sample 중 2-hop 이상에서 TR인 node pair 비율 확인
        
        Input:
            pos_src: [pos_N,]
            pos_dst: [pos_N,]
            TR_hop: [N+1,N+1] 
        """
        hop=TR_hop[pos_src,pos_dst]
        multi_hop_mask=hop>=2
        n_total=hop.numel()
        n_multi_hop=multi_hop_mask.sum().item()
        multi_hop_ratio=n_multi_hop/n_total
        return multi_hop_ratio

    @staticmethod
    def check_hard_negative_ratio_in_sample(
            neg_src:torch.Tensor,
            neg_dst:torch.Tensor,
            SR_label:torch.Tensor,
            SR_hop:torch.Tensor
        ):
        """
        negative sample 중 2-hop 이상의 static path로는 reachable 하지만, temporal path로는 unreachable한 node pair 비율 확인 

        Input:
            neg_src: [neg_N,]
            neg_dst: [neg_N,]
            SR_label: [N+1,N+1]
            SR_hop: [N+1,N+1]
        """
        sr_label=SR_label[neg_src,neg_dst]
        sr_hop=SR_hop[neg_src,neg_dst]
        SR_mask=(
            sr_label&
            (sr_hop>=2)
        )
        n_total=neg_src.numel()
        n_SR=SR_mask.sum().item()
        SR_ratio=n_SR/n_total
        return SR_ratio