import torch

class Metric:
    @staticmethod
    def compute_accuracy(
            pred_logit:torch.Tensor,
            label:torch.Tensor
        ):
        """
        Input:
            pred_logit: [B,]
            label: [B,]
        """
        pred=(torch.sigmoid(pred_logit)>=0.5).float()
        acc=(pred==label.float()).float().mean().item()
        return acc