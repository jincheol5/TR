import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import TrainUtils,Metric,EarlyStopper

class WalkModelTrainer:
    """
    Walk-based Model 학습/평가
    """
    @staticmethod
    def train_model(
            model:nn.Module,
            train_sample_loader:DataLoader,
            val_sample_loader:DataLoader,
            **kwargs
        ):
        """
        Train Skip-Gram
        """
        match kwargs["model_name"]:
            case "CTDNE":
                model.train_skipgram(
                    walk_len=kwargs["walk_len"],
                    min_walk_len=kwargs["min_walk_len"],
                    n_walk=kwargs["n_walk"],
                    n_window=kwargs["n_window"],
                    edge_sampling=kwargs["edge_sampling"],
                    neighbor_sampling=kwargs["neighbor_sampling"],
                    epoch=kwargs["walk_epoch"]
                )
            case "ATDGEB":
                model.train_skipgram(
                    k_list=kwargs["k_list"],
                    n_aggr=kwargs["n_aggr"],
                    min_points=kwargs["min_points"],
                    walk_len=kwargs["walk_len"],
                    n_lsbs=kwargs["n_lsbs"],
                    epoch=kwargs["walk_epoch"]
                )
        print(f"Finish to train Skip-Gram")

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
        Model Train
        """
        for epoch in tqdm(range(kwargs["epoch"]),desc=f"Model Training..."):
            model.train()
            for sample in tqdm(
                    train_sample_loader,
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
                label=torch.cat([
                    sample["pos_label"],
                    sample["neg_label"]
                ]).float()
                src=src.to(device)
                dst=dst.to(device)
                label=label.to(device)

                pred_logit=model(
                    src=src,
                    dst=dst
                ) # [n_sample,1]
                pred_logit=pred_logit.squeeze(-1) # -> [n_sample,]

                ### Loss
                criterion=nn.BCEWithLogitsLoss()
                loss=criterion(pred_logit,label)

                ### backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            """
            Validate Model
            """
            val_result=WalkModelTrainer.evaluate_model(
                model=model,
                sample_loader=val_sample_loader,
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

    @staticmethod
    def validate_model(
            model,
            val_sample_loader:list
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
        compute validate loss and acc
        """
        loss_list=[]
        acc_list=[]
        with torch.no_grad():
            for sample in tqdm(val_sample_loader,desc=f"Compute Val_Loss..."):
                src=torch.cat([
                    sample["pos_src"],
                    sample["neg_src"]
                ])
                dst=torch.cat([
                    sample["pos_dst"],
                    sample["neg_dst"]
                ])
                label=torch.cat([
                    sample["pos_label"],
                    sample["neg_label"]
                ]).float()
                src=src.to(device)
                dst=dst.to(device)
                label=label.to(device)

                pred_logit=model(
                    src=src,
                    dst=dst
                ) # [n_sample,1]
                pred_logit=pred_logit.squeeze(-1) # -> [n_sample,]

                ### Loss
                criterion=nn.BCEWithLogitsLoss()
                batch_loss=criterion(pred_logit,label)
                loss_list.append(batch_loss)

                ### compute ACC
                batch_acc=Metric.compute_accuracy(
                    pred_logit=pred_logit,
                    label=label
                )
                acc_list.append(batch_acc)
        return {
            "loss":torch.stack(loss_list).mean().item(),
            "acc":sum(acc_list)/len(acc_list)
        }

    @staticmethod
    def evaluate_model(
            model:nn.Module,
            test_sample_loader:list,
            **kwargs
        ):
        """
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model=model.to(device)
        model.eval()

        acc_list=[]
        with torch.no_grad():
            for sample in tqdm(
                    test_sample_loader,
                    desc=f"Evaluating..."
                ):
                src=torch.cat([
                    sample["pos_src"],
                    sample["neg_src"]
                ])
                dst=torch.cat([
                    sample["pos_dst"],
                    sample["neg_dst"]
                ])
                label=torch.cat([
                    sample["pos_label"],
                    sample["neg_label"]
                ]).float()
                src=src.to(device)
                dst=dst.to(device)
                label=label.to(device)

                pred_logit=model(
                    src=src,
                    dst=dst
                ) # [n_sample,1]
                pred_logit=pred_logit.squeeze(-1) # -> [n_sample,]

                ### compute ACC
                batch_acc=Metric.compute_accuracy(
                    pred_logit=pred_logit,
                    label=label
                )
                acc_list.append(batch_acc)
        return sum(acc_list)/len(acc_list)