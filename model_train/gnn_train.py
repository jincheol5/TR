import torch
import torch.nn as nn
from itertools import chain
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import TrainUtils,Metric,EarlyStopper

class GNNModelTrainer:
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
            ### Epoch마다 Memory-based model의 경우 memory state 초기화
            if kwargs["model_name"] in ("TGN"):
                model.memory.init_memory_state()
            model.train()
            for batch_event in tqdm(
                    train_loader,
                    desc=f"Training epoch {epoch+1}..."
                ):
                ### Eventstream
                event_src,event_dst,event_t,event_edge=batch_event
                if kwargs["model_name"] in ("TGN"):
                    event_src=event_src.to(device)
                    event_dst=event_dst.to(device)
                    event_t=event_t.to(device)
                    event_edge=event_edge.to(device)
                    model.update_model_memory(
                        src=event_src,
                        dst=event_dst,
                        event_t=event_t,
                        edge=event_edge
                    )

                ### TR Sample
                query_time=event_t.max().item()
                TR_sample=model.graph.random_TR_sampling(
                    n_sample=kwargs["n_batch_sample"], 
                    n_pair=kwargs["n_batch_pair"],
                    max_hop=kwargs["max_hop"],
                    query_time=query_time
                )
                src=TR_sample["src"]
                dst=TR_sample["dst"]
                query_t=TR_sample["query_t"]
                label=TR_sample["label"]
                src=src.to(device)
                dst=dst.to(device)
                query_t=query_t.to(device)
                label=label.to(device)

                pred_logit=model(
                    src=src,
                    dst=dst,
                    event_t=query_t
                ) # [B,1]

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

    @staticmethod
    def validate_model(
            model:nn.Module,
            val_loader:DataLoader,
            val_sample_loader:DataLoader,
            **kwargs
        ):
        """
        Validate Loss and ACC 계산
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model.to(device)
        model.graph.to_device(device=device)
        model.eval()

        """
        compute validate loss and acc
        """
        loss_list=[]
        acc_list=[]
        with torch.no_grad():
            if kwargs["model_name"] in ("TGN"):
                for src,dst,event_t,edge in tqdm(
                        val_loader,
                        desc=f"검증 eventstream에 대한 memory update 수행..."
                    ):
                    src=src.to(device)
                    dst=dst.to(device)
                    event_t=event_t.to(device)
                    edge=edge.to(device)
                    model.update_model_memory(
                        src=src,
                        dst=dst,
                        event_t=event_t,
                        edge=edge
                    )
            
            for sample in tqdm(val_sample_loader,desc=f"Compute Val_Loss..."):
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

                pred_logit=model(
                    src=src,
                    dst=dst,
                    event_t=query_t
                ) # [B,1]

                ### Loss
                pred_logit=pred_logit.squeeze(-1) # -> [B,]
                criterion=nn.BCEWithLogitsLoss()
                batch_loss=criterion(pred_logit,label)
                loss_list.append(batch_loss)

                ### ACC
                pred_logit=pred_logit.squeeze(-1) # -> [B,]
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
            test_loader:DataLoader,
            test_sample_loader:list,
            **kwargs
        ):
        """
        Memory-based GNN의 경우 evaluate_model 전에 val_loader에 대해 memory update 수행되어야 함.
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model.to(device)
        model.graph.to_device(device=device)
        model.eval()

        acc_list=[]
        with torch.no_grad():
            if kwargs["model_name"] in ("TGN"):
                for src,dst,event_t,edge in tqdm(
                        test_loader,
                        desc=f"평가 eventstream에 대한 memory update 수행..."
                    ):
                    src=src.to(device)
                    dst=dst.to(device)
                    event_t=event_t.to(device)
                    edge=edge.to(device)
                    model.update_model_memory(
                        src=src,
                        dst=dst,
                        event_t=event_t,
                        edge=edge
                    )

            for sample in tqdm(test_sample_loader,desc=f"Evaluating..."):
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

                pred_logit=model(
                    src=src,
                    dst=dst,
                    event_t=query_t
                ) # [B,1]

                ### ACC
                pred_logit=pred_logit.squeeze(-1) # -> [B,]
                batch_acc=Metric.compute_accuracy(
                    pred_logit=pred_logit,
                    label=label
                )
                acc_list.append(batch_acc)
        return sum(acc_list)/len(acc_list)