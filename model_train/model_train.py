import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import TrainUtils,Metric

class ModelTrainer:
    @staticmethod
    def train_walk_model(
            model:nn.Module,
            train_loader:DataLoader,
            val_sample_loader:list,
            **kwargs
        ):
        """
        Train skip-gram
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
                k_list=[2,4,6,8,10]
                model.train_skipgram(
                    k_list=k_list,
                    L=kwargs["L"],
                    min_points=kwargs["min_points"],
                    walk_len=kwargs["walk_len"],
                    n_sampling=kwargs["n_sampling"],
                    epoch=kwargs["walk_epoch"],
                    seed=kwargs["seed"]
                )
        print(f"finish train skip-gram")

        """
        Train decoder
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
        model train
        """
        for epoch in tqdm(range(kwargs["epoch"]),desc=f"Model Training..."):
            model.train()
            sample_loader=TrainUtils.get_TR_sample_loader(
                n_sample=kwargs["n_sample"],
                n_pair=kwargs["n_pair"],
                max_hop=kwargs["max_hop"],
                data_loader=train_loader,
                graph=model.graph
            )
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
            validate model
            """
            ModelTrainer.evaluate_walk_model(
                model=model,
                sample_loader=val_sample_loader,
                **kwargs
            )

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
        print(f"ACC: {sum(acc_list)/len(acc_list)}")