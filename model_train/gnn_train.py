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
    def train_model(
            model:nn.Module,
            train_loader:DataLoader,
            val_loader:DataLoader,
            val_sample_list:list,
            TR_label:torch.Tensor,
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

            ### Epoch마다 train_sample_list 생성
            train_sample_list=TrainUtils.get_fine_grained_TR_sample_list(
                n_pair=kwargs["n_pair"],
                data_loader=train_loader,
                TR_label=TR_label
            )

            model.train()
            for batch_event,batch_sample in tqdm(
                    zip(train_loader,train_sample_list),
                    total=len(train_sample_list),
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
                src=batch_sample["src"]
                dst=batch_sample["dst"]
                query_t=batch_sample["query_t"]
                label=batch_sample["label"]
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

                ### memory detach
                if kwargs["model_name"] in ("TGN"):
                    model.memory.memory_detach()

            """
            Validate Model
            """
            match kwargs["evaluate_type"]:
                case "coarse_grained":
                    val_result=GNNModelTrainer.validate_model_for_coarse_grained_TR(
                        model=model,
                        val_loader=val_loader,
                        val_sample_list=val_sample_list,
                        **kwargs
                    )
                case "fine_grained":
                    val_result=GNNModelTrainer.validate_model_for_fine_grained_TR(
                        model=model,
                        val_loader=val_loader,
                        val_sample_list=val_sample_list,
                        **kwargs
                    )
            val_acc=val_result["acc"]
            print(f"Validate ACC ({kwargs['evaluate_type']}): {val_acc}")

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
    def validate_model_for_coarse_grained_TR(
            model:nn.Module,
            val_loader:DataLoader,
            val_sample_list:DataLoader,
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

        loss_list=[]
        acc_list=[]
        with torch.no_grad():
            if kwargs["model_name"] in ("TGN"):
                for event_src,event_dst,event_t,event_edge in tqdm(
                        val_loader,
                        desc=f"val eventstream에 대한 memory update 수행..."
                    ):
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

            for src,dst,query_t,label in tqdm(val_sample_list,desc=f"Evaluating..."):
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
    def validate_model_for_fine_grained_TR(
            model:nn.Module,
            val_loader:DataLoader,
            val_sample_list:DataLoader,
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
            for batch_event,batch_sample in tqdm(
                    zip(val_loader,val_sample_list),
                    total=len(val_sample_list),
                    desc=f"Validate..."
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
                src=batch_sample["src"]
                dst=batch_sample["dst"]
                query_t=batch_sample["query_t"]
                label=batch_sample["label"]
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
    def evaluate_model_for_coarse_grained_TR(
            model:nn.Module,
            val_loader:DataLoader,
            test_loader:DataLoader,
            test_sample_list:list,
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

        """
        update memory for validate eventstream
        """
        if kwargs["model_name"] in ("TGN"):
            for event_src,event_dst,event_t,event_edge in tqdm(
                    val_loader,
                    desc=f"validate eventstream에 대한 memory update 수행..."
                ):
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

        """
        compute test acc
        """
        acc_list=[]
        with torch.no_grad():
            if kwargs["model_name"] in ("TGN"):
                for event_src,event_dst,event_t,event_edge in tqdm(
                        test_loader,
                        desc=f"test eventstream에 대한 memory update 수행..."
                    ):
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

            for src,dst,query_t,label in tqdm(test_sample_list,desc=f"Evaluating..."):
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

    @staticmethod
    def evaluate_model_for_fine_grained_TR(
            model:nn.Module,
            val_loader:DataLoader,
            test_loader:DataLoader,
            test_sample_list:list,
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

        """
        update memory for validate eventstream
        """
        if kwargs["model_name"] in ("TGN"):
            for event_src,event_dst,event_t,event_edge in tqdm(
                    val_loader,
                    desc=f"validate eventstream에 대한 memory update 수행..."
                ):
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

        """
        compute test acc
        """
        acc_list=[]
        with torch.no_grad():
            for batch_event,batch_sample in tqdm(
                    zip(test_loader,test_sample_list),
                    total=len(test_sample_list),
                    desc=f"Compute Test Acc..."
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
                src=batch_sample["src"]
                dst=batch_sample["dst"]
                query_t=batch_sample["query_t"]
                label=batch_sample["label"]
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