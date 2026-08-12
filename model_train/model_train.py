import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

class ModelTrainer:
    @staticmethod
    def train_walk_model(
            model:nn.Module,
            train_loader:DataLoader,
            val_loader:DataLoader,
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
                    n_context_window=kwargs["n_context_window"],
                    max_attempt=kwargs["max_attempt"],
                    edge_sampling=kwargs["edge_sampling"],
                    neighbor_sampling=kwargs["neighbor_sampling"],
                    epoch=kwargs["walk_epoch"],
                    seed=kwargs["seed"]
                )
            case "ATDGEB":
                k_list=[2,4,6,8,10]
                model.train_skipgram(
                    k_list=k_list,
                    L=kwargs["L"],
                    min_points=kwargs["min_points"],
                    max_walk_len=kwargs["max_walk_len"],
                    n_sampling=kwargs["n_sampling"],
                    epoch=kwargs["walk_epoch"],
                    seed=kwargs["seed"]
                )

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
            for src,dst,event_t,edge_idx in tqdm(
                    train_loader,
                    desc=f"Training epoch: {epoch+1}..."
                ):
                src=src.to(device) # [B,]
                dst=dst.to(device) # [B,]
                event_t=event_t.to(device) # [B,]
                edge_idx=edge_idx.to(device) # [B,]

                # TR sampling
                query_time=event_t[-1].item()

    @staticmethod
    def evaluate_walk(
            model:nn.Module,
            data_loader:DataLoader,
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
        model.graph.set_random_seed(kwargs["seed"])