import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import TrainUtils,Metric,EarlyStopper,ReaCH_TGN_Utils

class ReaCH_TGN_Trainer:
    """
    GNN Model 학습/평가
    """
    @staticmethod
    def train_model(
            model:nn.Module,
            train_loader:DataLoader,
            val_loader:DataLoader,
            val_sample_loader:DataLoader,
            **kwargs
        ):
        """
        Set GPU, Optimizer
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model=model.to(device)
        model.graph.to_device(device=device)

        if kwargs["optimizer"]=="adam":
            optimizer=torch.optim.Adam(
                model.parameters(),
                lr=kwargs["lr"]
            )
        else:
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=kwargs["lr"]
            )

        """
        Set Early Stopper
        """
        if kwargs["early_stop"]:
            early_stop=EarlyStopper(patience=kwargs["patience"])

        """
        Model train
        """
        for epoch in tqdm(range(kwargs["epoch"]),desc=f"Model Training..."):
            model.train()
            for src,dst,event_t,edge in tqdm(
                    train_loader,
                    desc=f"Training epoch: {epoch+1}..."
                ):
                src=src.to(device)
                dst=dst.to(device)
                event_t=event_t.to(device)
                label=label.to(device)

                ### 1. Temporal Augmentation and Contrastive Learning
                view_A_event_t=ReaCH_TGN_Utils.Temporal_Augmentation(
                    event_t=event_t,
                    jitter_std=0.01,
                    jitter_range=0.01
                )
                view_B_event_t=ReaCH_TGN_Utils.Temporal_Augmentation(
                    event_t=event_t,
                    jitter_std=0.01,
                    jitter_range=0.01
                )

                model.update_model_memory(
                    src=src,
                    dst=dst,
                    event_t=view_A_event_t,
                    edge=edge
                )



                

                ### Loss
                pred_logit=pred_logit.squeeze(-1) # -> [B,]
                criterion=nn.BCEWithLogitsLoss()
                loss=criterion(pred_logit,label)

                ### backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            """
            Validate Model
            """
            val_result=GNNModelTrainer.validate_model(
                model=model,
                val_loader=val_loader,
                val_sample_loader=val_sample_loader,
                **kwargs
            )
            val_acc=val_result["acc"]
            print(f"Validate ACC: {val_acc}")

            """
            Check Early Stop
            """
            val_loss=val_result["loss"]
            print(f"{epoch+1} epoch Validate Loss: {val_loss}")
            if kwargs["early_stop"]:
                pre_model=early_stop(
                    val_loss=val_loss,
                    model=model
                )
                if early_stop.early_stop:
                    model=pre_model
                    print(f"Early Stop in epoch {epoch+1}")
                    break
        return model