import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import TrainUtils,Metric,EarlyStopper

class GNNModelTrainer:
    """
    GNN Model 학습/평가
    """
    @staticmethod
    def train_gnn_model(
            model:nn.Module,
            train_loader:DataLoader,
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
            ### generate train sample list
            sample_list=TrainUtils.get_sample_list(
                graph=model.graph,
                data_loader=train_loader,
                max_hop=kwargs["max_hop"],
                n_pair=kwargs["n_pair"],
                n_batch_sample=kwargs["n_batch_sample"]
            )

            model.train()
            for (event_src,event_dst,event_t,event_edge),sample in tqdm(
                    zip(train_loader,sample_list),
                    total=len(sample_list),
                    desc=f"Training epoch: {epoch+1}..."
                ):

                src=torch.cat([
                    sample["pos_src"],
                    sample["neg_src"]
                ])
                dst=torch.cat([
                    sample["pos_dst"],
                    sample["neg_dst"]
                ])
                query_t=torch.cat([
                    sample["pos_pair_t"],
                    sample["neg_pair_t"]
                ]).float()
                label=torch.cat([
                    sample["pos_label"],
                    sample["neg_label"]
                ]).float()
                src=src.to(device)
                dst=dst.to(device)
                query_t=query_t.to(device)
                label=label.to(device)

                match kwargs["model_name"]:
                    case "TGN":
                        event_src=event_src.to(device)
                        event_dst=event_dst.to(device)
                        event_t=event_t.to(device)
                        event_edge=event_edge.to(device)
                        event={
                            "src":event_src,
                            "dst":dst,
                            "event_t":event_t,
                            "edge":event_edge
                        }
                        pred_logit=model(
                            src=src,
                            dst=dst,
                            query_t=query_t,
                            event=event
                        ) # [B,1]

                    case "TGAT"|"DyGFormer":
                        pred_logit=model(
                            src=src,
                            dst=dst,
                            query_t=query_t
                        ) # [B,1]

                ### Loss
                pred_logit=pred_logit.squeeze(-1) # -> [B,]
                criterion=nn.BCEWithLogitsLoss()
                loss=criterion(pred_logit,label)

                ### backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


    @staticmethod
    def compute_validate_loss(
            model,
            val_loader:DataLoader,
            val_sample_loader:list,
            **kwargs
        ):
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model.to(device)
        model.eval()

        """
        compute validate loss
        """
        with torch.no_grad():
            if kwargs["model_name"] in ("TGN"):
                for event_src,event_dst,event_t,event_edge in tqdm(
                        val_loader,
                        desc=f"update memory..."
                    ):
                    event_src=event_src.to(device)
                    event_dst=event_dst.to(device)
                    event_t=event_t.to(device)
                    event_edge=event_edge.to(device)
                    event={
                        "src":event_src,
                        "dst":dst,
                        "event_t":event_t,
                        "edge":event_edge
                    }
                    model(event=event) # memory update만 수행
                

        loss_list=[]
        with torch.no_grad():
            for (event_src,event_dst,event_t,event_edge),sample in tqdm(
                    zip(val_loader,val_sample_loader),
                    total=len(val_loader),
                    desc=f"Compute Val_Loss..."
                ):
                src=torch.cat([
                    sample["pos_src"],
                    sample["neg_src"]
                ])
                dst=torch.cat([
                    sample["pos_dst"],
                    sample["neg_dst"]
                ])
                query_t=torch.cat([
                    sample["pos_pair_t"],
                    sample["neg_pair_t"]
                ]).float()
                label=torch.cat([
                    sample["pos_label"],
                    sample["neg_label"]
                ]).float()
                src=src.to(device)
                dst=dst.to(device)
                query_t=query_t.to(device)
                label=label.to(device)

                match kwargs["model_name"]:
                    case "TGN":
                        event_src=event_src.to(device)
                        event_dst=event_dst.to(device)
                        event_t=event_t.to(device)
                        event_edge=event_edge.to(device)
                        event={
                            "src":event_src,
                            "dst":dst,
                            "event_t":event_t,
                            "edge":event_edge
                        }
                        pred_logit=model(
                            src=src,
                            dst=dst,
                            query_t=query_t,
                            event=event
                        ) # [B,1]

                    case "TGAT"|"DyGFormer":
                        pred_logit=model(
                            src=src,
                            dst=dst,
                            query_t=query_t
                        ) # [B,1]

                ### Loss
                pred_logit=pred_logit.squeeze(-1) # -> [B,]
                criterion=nn.BCEWithLogitsLoss()
                loss=criterion(pred_logit,label)
                loss_list.append(loss)
        return torch.stack(loss_list).mean().item()