import argparse
import torch
from torch.utils.data import DataLoader
from utils import DataUtils,TrainUtils,TemporalGraphDataset
from graph import TGN_Graph,DyGFormer_Graph
from model import TGAT,TGN,DyGFormer,ReaCH_TGN
from model_train import GNNModelTrainer,ReaCH_TGN_Trainer

"""
<< Test >> 
model_train.ModelTrainer

Test Model:
    - TGAT
    - TGN
    - DyGFormer
"""
def test_fn(**kwargs):
    match kwargs["model_name"]:
        case "TGAT":
            """
            Test Model: TGAT
            """
            data=DataUtils.preprocess_graph_dataset(dataset_name=f"enron")
            graph_df=data["graph_df"]
            node_ft=data["node_ft"]
            edge_ft=data["edge_ft"]
            node_dim=data["node_dim"]
            edge_dim=data["edge_dim"]
            graph=TGN_Graph(
                graph_df=graph_df,
                node_ft=node_ft,
                edge_ft=edge_ft,
                node_dim=node_dim,
                edge_dim=edge_dim
            )
            seed=1
            graph.set_random_seed(seed=seed)

            ### TR sample 관련 파라미터
            batch_size=200
            max_hop=5
            n_sample=1000
            n_pair=10
            evaluate_type=kwargs["evaluate_type"]

            ### 모델 관련 파라미터
            n_layer=2
            n_neighbor=10
            n_head=4

            ### 학습 관련 파라미터
            time_dim=32
            latent_dim=32
            embed_dim=32
            epoch=100
            lr=0.0005
            optimizer=f"adam"
            early_stop=True
            patience=10

            ### model config
            model_config={
                "model_name":kwargs["model_name"],
                "seed":seed,
                "batch_size":batch_size,
                "max_hop":max_hop,
                "n_sample":n_sample,
                "n_pair":n_pair,
                "evaluate_type":evaluate_type,
                "n_layer":n_layer,
                "n_neighbor":n_neighbor,
                "n_head":n_head,
                "time_dim":time_dim,
                "latent_dim":latent_dim,
                "embed_dim":embed_dim,
                "epoch":epoch,
                "lr":lr,
                "optimizer":optimizer,
                "early_stop":early_stop,
                "patience":patience
            }

            ### set data_loader
            train_df,val_df,test_df=TrainUtils.split_graph_df(df=graph_df)
            val_query_time=val_df["t"].max()
            test_query_time=test_df["t"].max()
            train_dataset=TemporalGraphDataset(df=train_df)
            val_dataset=TemporalGraphDataset(df=val_df)
            test_dataset=TemporalGraphDataset(df=test_df)
            train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
            val_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
            test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

            ### set sample_list
            train_TR_result=DataUtils.load_TR_result(
                dataset_name=f"enron",
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="train"
            )
            train_TR_label=train_TR_result["TR_label"]
            val_TR_result=DataUtils.load_TR_result(
                dataset_name=f"enron",
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="val"
            )
            val_TR_label=val_TR_result["TR_label"]
            test_TR_result=DataUtils.load_TR_result(
                dataset_name=f"enron",
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="test"
            )
            test_TR_label=test_TR_result["TR_label"]
            if evaluate_type=="coarse_grained":
                n_node=graph.get_num_node()
                val_sample_list=TrainUtils.get_coarse_grained_TR_sample_list(
                    n_node=n_node,
                    n_sample=n_sample,
                    n_pair=n_pair,
                    query_time=val_query_time,
                    batch_size=batch_size,
                    TR_label=val_TR_label
                )
                test_sample_list=TrainUtils.get_fine_grained_TR_sample_list(
                    n_node=n_node,
                    n_sample=n_sample,
                    n_pair=n_pair,
                    query_time=test_query_time,
                    batch_size=batch_size,
                    TR_label=test_TR_label
                )
            else: # fine_grained
                val_sample_list=TrainUtils.get_fine_grained_TR_sample_list(
                    n_pair=n_pair,
                    data_loader=val_loader,
                    TR_label=val_TR_label
                )
                test_sample_list=TrainUtils.get_fine_grained_TR_sample_list(
                    n_pair=n_pair,
                    data_loader=test_loader,
                    TR_label=test_TR_label
                )

            ### set model and train
            model=TGAT(
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                latent_dim=latent_dim,
                embed_dim=embed_dim,
                graph=graph,
                n_layer=n_layer,
                n_neighbor=n_neighbor,
                n_head=n_head
            )
            model=GNNModelTrainer.train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                val_sample_list=val_sample_list,
                TR_label=train_TR_label,
                **model_config
            )
            acc=GNNModelTrainer.evaluate_model_for_fine_grained_TR(
                model=model,
                val_loader=val_loader,
                test_loader=test_loader,
                test_sample_list=test_sample_list,
                **model_config
            )
            print(f"Evaluate ACC: {acc}")

        case "TGN":
            data=DataUtils.preprocess_graph_dataset(dataset_name=f"enron")
            graph_df=data["graph_df"]
            node_ft=data["node_ft"]
            edge_ft=data["edge_ft"]
            node_dim=data["node_dim"]
            edge_dim=data["edge_dim"]
            graph=TGN_Graph(
                graph_df=graph_df,
                node_ft=node_ft,
                edge_ft=edge_ft,
                node_dim=node_dim,
                edge_dim=edge_dim
            )
            seed=1
            graph.set_random_seed(seed=seed)

            ### TR sample 관련 파라미터
            batch_size=200
            max_hop=5
            n_sample=1000
            n_pair=10
            evaluate_type=kwargs["evaluate_type"]

            ### 모델 관련 파라미터
            n_layer=2
            n_neighbor=10
            n_head=4

            ### 학습 관련 파라미터
            time_dim=32
            latent_dim=32
            msg_dim=32
            mem_dim=32
            embed_dim=32
            epoch=100
            lr=0.0005
            optimizer=f"adam"
            early_stop=True
            patience=5

            ### model config
            model_config={
                "model_name":kwargs["model_name"],
                "seed":seed,
                "batch_size":batch_size,
                "max_hop":max_hop,
                "n_sample":n_sample,
                "n_pair":n_pair,
                "evaluate_type":evaluate_type,
                "n_layer":n_layer,
                "n_neighbor":n_neighbor,
                "n_head":n_head,
                "time_dim":time_dim,
                "latent_dim":latent_dim,
                "msg_dim":msg_dim,
                "mem_dim":mem_dim,
                "embed_dim":embed_dim,
                "epoch":epoch,
                "lr":lr,
                "optimizer":optimizer,
                "early_stop":early_stop,
                "patience":patience
            }

            ### set data_loader
            train_df,val_df,test_df=TrainUtils.split_graph_df(df=graph_df)
            val_query_time=val_df["t"].max()
            test_query_time=test_df["t"].max()
            train_dataset=TemporalGraphDataset(df=train_df)
            val_dataset=TemporalGraphDataset(df=val_df)
            test_dataset=TemporalGraphDataset(df=test_df)
            train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
            val_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
            test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

            ### set sample_list
            train_TR_result=DataUtils.load_TR_result(
                dataset_name=f"enron",
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="train"
            )
            train_TR_label=train_TR_result["TR_label"]
            val_TR_result=DataUtils.load_TR_result(
                dataset_name=f"enron",
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="val"
            )
            val_TR_label=val_TR_result["TR_label"]
            test_TR_result=DataUtils.load_TR_result(
                dataset_name=f"enron",
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="test"
            )
            test_TR_label=test_TR_result["TR_label"]
            if evaluate_type=="coarse_grained":
                n_node=graph.get_num_node()
                val_sample_list=TrainUtils.get_coarse_grained_TR_sample_list(
                    n_node=n_node,
                    n_sample=n_sample,
                    n_pair=n_pair,
                    query_time=val_query_time,
                    batch_size=batch_size,
                    TR_label=val_TR_label
                )
                test_sample_list=TrainUtils.get_fine_grained_TR_sample_list(
                    n_node=n_node,
                    n_sample=n_sample,
                    n_pair=n_pair,
                    query_time=test_query_time,
                    batch_size=batch_size,
                    TR_label=test_TR_label
                )
            else: # fine_grained
                val_sample_list=TrainUtils.get_fine_grained_TR_sample_list(
                    n_pair=n_pair,
                    data_loader=val_loader,
                    TR_label=val_TR_label
                )
                test_sample_list=TrainUtils.get_fine_grained_TR_sample_list(
                    n_pair=n_pair,
                    data_loader=test_loader,
                    TR_label=test_TR_label
                )

            ### set model and train
            model=TGN(
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                latent_dim=latent_dim,
                msg_dim=msg_dim,
                mem_dim=mem_dim,
                embed_dim=embed_dim,
                graph=graph,
                n_layer=n_layer,
                n_neighbor=n_neighbor,
                n_head=n_head
            )   
            model=GNNModelTrainer.train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                val_sample_list=val_sample_list,
                TR_label=train_TR_label,
                **model_config
            )
            acc=GNNModelTrainer.evaluate_model_for_fine_grained_TR(
                model=model,
                val_loader=val_loader,
                test_loader=test_loader,
                test_sample_list=test_sample_list,
                **model_config
            )
            print(f"Evaluate ACC: {acc}")

        case "DyGFormer":
            data=DataUtils.preprocess_graph_dataset(dataset_name=f"enron")
            graph_df=data["graph_df"]
            node_ft=data["node_ft"]
            edge_ft=data["edge_ft"]
            node_dim=data["node_dim"]
            edge_dim=data["edge_dim"]
            graph=DyGFormer_Graph(
                graph_df=graph_df,
                node_ft=node_ft,
                edge_ft=edge_ft,
                node_dim=node_dim,
                edge_dim=edge_dim
            )
            seed=1
            graph.set_random_seed(seed=seed)

            ### TR sample 관련 파라미터
            batch_size=200
            max_hop=5
            n_sample=1000
            n_pair=10
            evaluate_type=kwargs["evaluate_type"]

            ### 모델 관련 파라미터
            n_layer=2
            n_neighbor=10
            n_head=4
            max_seq_len=10
            patch_size=5

            ### 학습 관련 파라미터
            latent_dim=32
            time_dim=32
            co_dim=32
            common_dim=32
            embed_dim=32
            epoch=100
            lr=0.0005
            optimizer=f"adam"
            early_stop=True
            patience=5

            ### model config
            model_config={
                "model_name":kwargs["model_name"],
                "seed":seed,
                "batch_size":batch_size,
                "max_hop":max_hop,
                "n_sample":n_sample,
                "n_pair":n_pair,
                "evaluate_type":evaluate_type,
                "n_layer":n_layer,
                "n_neighbor":n_neighbor,
                "n_head":n_head,
                "max_seq_len":max_seq_len,
                "patch_size":patch_size,
                "latent_dim":latent_dim,
                "time_dim":time_dim,
                "co_dim":co_dim,
                "common_dim":common_dim,
                "embed_dim":embed_dim,
                "epoch":epoch,
                "lr":lr,
                "optimizer":optimizer,
                "early_stop":early_stop,
                "patience":patience
            }

            ### set data_loader
            train_df,val_df,test_df=TrainUtils.split_graph_df(df=graph_df)
            val_query_time=val_df["t"].max()
            test_query_time=test_df["t"].max()
            train_dataset=TemporalGraphDataset(df=train_df)
            val_dataset=TemporalGraphDataset(df=val_df)
            test_dataset=TemporalGraphDataset(df=test_df)
            train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
            val_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
            test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

            ### set sample_list
            train_TR_result=DataUtils.load_TR_result(
                dataset_name=f"enron",
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="train"
            )
            train_TR_label=train_TR_result["TR_label"]
            val_TR_result=DataUtils.load_TR_result(
                dataset_name=f"enron",
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="val"
            )
            val_TR_label=val_TR_result["TR_label"]
            test_TR_result=DataUtils.load_TR_result(
                dataset_name=f"enron",
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="test"
            )
            test_TR_label=test_TR_result["TR_label"]
            if evaluate_type=="coarse_grained":
                n_node=graph.get_num_node()
                val_sample_list=TrainUtils.get_coarse_grained_TR_sample_list(
                    n_node=n_node,
                    n_sample=n_sample,
                    n_pair=n_pair,
                    query_time=val_query_time,
                    batch_size=batch_size,
                    TR_label=val_TR_label
                )
                test_sample_list=TrainUtils.get_fine_grained_TR_sample_list(
                    n_node=n_node,
                    n_sample=n_sample,
                    n_pair=n_pair,
                    query_time=test_query_time,
                    batch_size=batch_size,
                    TR_label=test_TR_label
                )
            else: # fine_grained
                val_sample_list=TrainUtils.get_fine_grained_TR_sample_list(
                    n_pair=n_pair,
                    data_loader=val_loader,
                    TR_label=val_TR_label
                )
                test_sample_list=TrainUtils.get_fine_grained_TR_sample_list(
                    n_pair=n_pair,
                    data_loader=test_loader,
                    TR_label=test_TR_label
                )

            ### set model and train
            model=DyGFormer(
                node_dim=node_dim,
                edge_dim=edge_dim,
                latent_dim=latent_dim,
                time_dim=time_dim,
                co_dim=co_dim,
                common_dim=common_dim,
                embed_dim=embed_dim,
                max_seq_len=max_seq_len,
                patch_size=patch_size,
                graph=graph,
                n_neighbor=n_neighbor,
                n_layer=n_layer,
                n_head=n_head
            )
            model=GNNModelTrainer.train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                val_sample_list=val_sample_list,
                TR_label=train_TR_label,
                **model_config
            )
            acc=GNNModelTrainer.evaluate_model_for_fine_grained_TR(
                model=model,
                val_loader=val_loader,
                test_loader=test_loader,
                test_sample_list=test_sample_list,
                **model_config
            )
            print(f"Evaluate ACC: {acc}")

        case "ReaCH-TGN":
            data=DataUtils.preprocess_graph_dataset(dataset_name=f"enron")
            graph_df=data["graph_df"]
            node_ft=data["node_ft"]
            edge_ft=data["edge_ft"]
            node_dim=data["node_dim"]
            edge_dim=data["edge_dim"]
            graph=TGN_Graph(
                graph_df=graph_df,
                node_ft=node_ft,
                edge_ft=edge_ft,
                node_dim=node_dim,
                edge_dim=edge_dim
            )
            seed=1
            graph.set_random_seed(seed=seed)

            ### TR sample 관련 파라미터
            batch_size=200
            max_hop=5
            n_sample=1000
            n_pair=10
            evaluate_type=kwargs["evaluate_type"]

            ### 모델 관련 파라미터
            n_layer=2
            n_neighbor=10
            n_head=4

            ### 학습 관련 파라미터
            time_dim=32
            latent_dim=32
            msg_dim=32
            mem_dim=32
            embed_dim=32
            epoch=100
            lr=0.0005
            optimizer=f"adam"
            early_stop=True
            patience=5

            ### model config
            model_config={
                "model_name":kwargs["model_name"],
                "seed":seed,
                "batch_size":batch_size,
                "max_hop":max_hop,
                "n_sample":n_sample,
                "n_pair":n_pair,
                "evaluate_type":evaluate_type,
                "n_layer":n_layer,
                "n_neighbor":n_neighbor,
                "n_head":n_head,
                "time_dim":time_dim,
                "latent_dim":latent_dim,
                "msg_dim":msg_dim,
                "mem_dim":mem_dim,
                "embed_dim":embed_dim,
                "epoch":epoch,
                "lr":lr,
                "optimizer":optimizer,
                "early_stop":early_stop,
                "patience":patience
            }

            ### set data_loader
            train_df,val_df,test_df=TrainUtils.split_graph_df(df=graph_df)
            val_query_time=val_df["t"].max()
            test_query_time=test_df["t"].max()
            train_dataset=TemporalGraphDataset(df=train_df)
            val_dataset=TemporalGraphDataset(df=val_df)
            test_dataset=TemporalGraphDataset(df=test_df)
            train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
            val_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
            test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

            ### set sample_list
            train_TR_result=DataUtils.load_TR_result(
                dataset_name=f"enron",
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="train"
            )
            val_TR_result=DataUtils.load_TR_result(
                dataset_name=f"enron",
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="val"
            )
            val_TR_label=val_TR_result["TR_label"]
            test_TR_result=DataUtils.load_TR_result(
                dataset_name=f"enron",
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="test"
            )
            test_TR_label=test_TR_result["TR_label"]
            if evaluate_type=="coarse_grained":
                n_node=graph.get_num_node()
                val_sample_list=TrainUtils.get_coarse_grained_TR_sample_list(
                    n_node=n_node,
                    n_sample=n_sample,
                    n_pair=n_pair,
                    query_time=val_query_time,
                    batch_size=batch_size,
                    TR_label=val_TR_label
                )
                test_sample_list=TrainUtils.get_fine_grained_TR_sample_list(
                    n_node=n_node,
                    n_sample=n_sample,
                    n_pair=n_pair,
                    query_time=test_query_time,
                    batch_size=batch_size,
                    TR_label=test_TR_label
                )
            else: # fine_grained
                val_sample_list=TrainUtils.get_fine_grained_TR_sample_list(
                    n_pair=n_pair,
                    data_loader=val_loader,
                    TR_label=val_TR_label
                )
                test_sample_list=TrainUtils.get_fine_grained_TR_sample_list(
                    n_pair=n_pair,
                    data_loader=test_loader,
                    TR_label=test_TR_label
                )

            ### set model and train
            model=ReaCH_TGN(
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                msg_dim=msg_dim,
                mem_dim=mem_dim,
                latent_dim=latent_dim,
                embed_dim=embed_dim,
                graph=graph,
                n_layer=n_layer,
                n_neighbor=n_neighbor,
                n_head=n_head
            )
            model=ReaCH_TGN_Trainer.train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                val_sample_list=val_sample_list,
                TR_result=train_TR_result,
                **model_config
            )
            acc=ReaCH_TGN_Trainer.evaluate_model_for_fine_grained_TR(
                model=model,
                val_loader=val_loader,
                test_loader=test_loader,
                test_sample_list=test_sample_list,
                **model_config
            )
            print(f"Evaluate ACC: {acc}")

if __name__=="__main__":
    """
    Execute test_fn
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",
        type=str,
        choices=["TGAT","TGN","DyGFormer","ReaCH-TGN"],
        default=f"TGAT"
    )
    parser.add_argument("--evaluate_type",
        type=str,
        choices=["coarse_grained","fine_grained"],
        default=f"coarse_grained"
    )
    args=parser.parse_args()
    test_config={
        "model_name":args.model_name,
        "evaluate_type":args.evaluate_type
    }
    test_fn(**test_config)