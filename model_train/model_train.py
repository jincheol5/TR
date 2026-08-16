import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import TrainUtils,Metric,EarlyStopper

class ModelTrainer:
    @staticmethod
    def train_walk_model(
            model:nn.Module,
            train_loader:DataLoader,
            val_sample_loader:list,
            **kwargs
        ):
        """
        Set Early Stopper
        """
        if kwargs["early_stop"]:
            early_stop=EarlyStopper(patience=kwargs["patience"])

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
        print(f"Finish to train Skip-Gram")

        """
        Train Decoder
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
        Model Train
        """
        for epoch in tqdm(range(kwargs["epoch"]),desc=f"Model Training..."):
            model.train()
            sample_loader=TrainUtils.get_sample_loader(
                n_sample=kwargs["n_sample"],
                n_pair=kwargs["n_pair"],
                max_hop=kwargs["max_hop"],
                data_loader=train_loader,
                graph=model.graph
            )
            TrainUtils.check_sample_loader(sample_loader=sample_loader)
            for sample in tqdm(
                    sample_loader,
                    desc=f"Training epoch: {epoch+1}..."
                ):
                src=sample["src"]
                dst=sample["dst"]
                label=sample["label"]
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
            acc=ModelTrainer.evaluate_walk_model(
                model=model,
                sample_loader=val_sample_loader,
                **kwargs
            )
            print(f"Validate ACC: {acc}")

            """
            Check Early Stop
            """
            val_loss=ModelTrainer.compute_validate_loss(
                model=model,
                val_sample_loader=val_sample_loader
            )
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
    def compute_validate_loss(
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
        compute validate loss
        """
        loss_list=[]
        with torch.no_grad():
            for sample in tqdm(val_sample_loader,desc=f"Compute Val_Loss..."):
                src=sample["src"]
                dst=sample["dst"]
                label=sample["label"]
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
                loss_list.append(loss)
        return torch.stack(loss_list).mean().item()

    @staticmethod
    def evaluate_walk_model(
            model:nn.Module,
            sample_loader:list,
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
                    sample_loader,
                    desc=f"Evaluating..."
                ):
                src=sample["src"]
                dst=sample["dst"]
                label=sample["label"]
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
        acc=sum(acc_list)/len(acc_list)
        return acc