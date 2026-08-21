import argparse
import torch
from torch.utils.data import DataLoader
from utils import DataUtils,TrainUtils,TemporalGraphDataset,TRDataset
from graph import TGN_Graph,DyGFormer_Graph
from model import TGAT,TGN,DyGFormer
from model_train import GNNModelTrainer

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
            # TR sample 관련 파라미터
            seed=1
            batch_size=200
            max_hop=5
            n_pair=10
            n_sample=1000
            n_batch_sample=100

            data=DataUtils.preprocess_graph(dataset_name=f"enron")
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
            graph.set_random_seed(seed=seed)

            ### set data_loader, sample_loader
            train_df,val_df,test_df=TrainUtils.split_graph_df(df=graph_df)
            val_query_time=val_df["t"].max()
            test_query_time=test_df["t"].max()

            train_dataset=TemporalGraphDataset(df=train_df)
            val_dataset=TemporalGraphDataset(df=val_df)
            test_dataset=TemporalGraphDataset(df=test_df)

            train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
            val_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
            test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

            val_sample=graph.random_TR_sampling(
                n_sample=n_sample,
                n_pair=n_pair,
                query_time=val_query_time,
                max_hop=max_hop
            )
            print(f"Finish to generate val sample!")

            test_sample=graph.random_TR_sampling(
                n_sample=n_sample,
                n_pair=n_pair,
                query_time=test_query_time,
                max_hop=max_hop
            )
            print(f"Finish to generate test sample!")

            # Dataset
            val_sample_dataset=TRDataset(sample=val_sample,query_time=val_query_time)
            test_sample_dataset=TRDataset(sample=test_sample,query_time=test_query_time)

            # DataLoader, query_time 기준으로 생성되었기 때문에 shuffle 가능
            val_sample_loader=DataLoader(dataset=val_sample_dataset,batch_size=batch_size,shuffle=True)
            test_sample_loader=DataLoader(dataset=test_sample_dataset,batch_size=batch_size,shuffle=True)

            ### 하이퍼 파라미터
            n_layer=2
            n_neighbor=10
            n_head=4

            # 모델 학습 관련
            latent_dim=32
            time_dim=32
            output_dim=32
            epoch=1
            lr=0.0005
            optimizer=f"adam"
            early_stop=True
            patience=5

            ### model config
            model_config={
                "seed":seed,
                "batch_size":batch_size,
                "max_hop":max_hop,
                "n_pair":n_pair,
                "n_sample":n_sample,
                "n_batch_sample":n_batch_sample,
                "model_name":kwargs["model_name"],
                "n_layer":n_layer,
                "n_neighbor":n_neighbor,
                "n_head":n_head,
                "latent_dim":latent_dim,
                "time_dim":time_dim,
                "output_dim":output_dim,
                "epoch":epoch,
                "lr":lr,
                "optimizer":optimizer,
                "early_stop":early_stop,
                "patience":patience
            }

            ### set model and train
            model=TGAT(
                node_dim=node_dim,
                edge_dim=edge_dim,
                latent_dim=latent_dim,
                time_dim=time_dim,
                output_dim=output_dim,
                graph=graph,
                n_layer=n_layer,
                n_neighbor=n_neighbor,
                n_head=n_head
            )
            model=GNNModelTrainer.train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                val_sample_loader=val_sample_loader,
                **model_config
            )
            acc=GNNModelTrainer.evaluate_model(
                model=model,
                test_loader=test_loader,
                test_sample_loader=test_sample_loader,
                **model_config
            )
            print(f"Evaluate ACC: {acc}")

        case "TGN":
            # TR sample 관련 파라미터
            seed=1
            batch_size=200
            max_hop=5
            n_pair=10
            n_sample=1000
            n_batch_sample=100

            data=DataUtils.preprocess_graph(dataset_name=f"enron")
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
            graph.set_random_seed(seed=seed)

            ### set data_loader, sample_loader
            train_df,val_df,test_df=TrainUtils.split_graph_df(df=graph_df)
            val_query_time=val_df["t"].max()
            test_query_time=test_df["t"].max()

            train_dataset=TemporalGraphDataset(df=train_df)
            val_dataset=TemporalGraphDataset(df=val_df)
            test_dataset=TemporalGraphDataset(df=test_df)

            train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
            val_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
            test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

            val_sample=graph.random_TR_sampling(
                n_sample=n_sample,
                n_pair=n_pair,
                query_time=val_query_time,
                max_hop=max_hop
            )
            print(f"Finish to generate val sample!")

            test_sample=graph.random_TR_sampling(
                n_sample=n_sample,
                n_pair=n_pair,
                query_time=test_query_time,
                max_hop=max_hop
            )
            print(f"Finish to generate test sample!")

            # Dataset
            val_sample_dataset=TRDataset(sample=val_sample,query_time=val_query_time)
            test_sample_dataset=TRDataset(sample=test_sample,query_time=test_query_time)

            # DataLoader, query_time 기준으로 생성되었기 때문에 shuffle 가능
            val_sample_loader=DataLoader(dataset=val_sample_dataset,batch_size=batch_size,shuffle=True)
            test_sample_loader=DataLoader(dataset=test_sample_dataset,batch_size=batch_size,shuffle=True)

            ### 하이퍼 파라미터
            n_layer=2
            n_neighbor=10
            n_head=4
            msg_fn=f"concat"
            aggr_fn=f"last"

            # 모델 학습 관련
            mem_dim=32
            msg_dim=32
            latent_dim=32
            time_dim=32
            output_dim=32
            epoch=1
            lr=0.0005
            optimizer=f"adam"
            early_stop=True
            patience=5

            ### model config
            model_config={
                "seed":seed,
                "batch_size":batch_size,
                "max_hop":max_hop,
                "n_pair":n_pair,
                "n_sample":n_sample,
                "n_batch_sample":n_batch_sample,
                "model_name":kwargs["model_name"],
                "n_layer":n_layer,
                "n_neighbor":n_neighbor,
                "n_head":n_head,
                "latent_dim":latent_dim,
                "time_dim":time_dim,
                "output_dim":output_dim,
                "epoch":epoch,
                "lr":lr,
                "optimizer":optimizer,
                "early_stop":early_stop,
                "patience":patience
            }

            ### set model and train
            model=TGN(
                node_dim=node_dim,
                edge_dim=edge_dim,
                mem_dim=mem_dim,
                latent_dim=latent_dim,
                msg_dim=msg_dim,
                time_dim=time_dim,
                output_dim=output_dim,
                graph=graph,
                n_layer=n_layer,
                n_neighbor=n_neighbor,
                n_head=n_head,
                msg_fn=msg_fn,
                aggr_fn=aggr_fn
            )
            model=GNNModelTrainer.train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                val_sample_loader=val_sample_loader,
                **model_config
            )
            # 평가 전에 val_loader에 대해 memory update
            model=GNNModelTrainer.update_model_memory(
                model=model,
                data_loader=val_loader
            )
            acc=GNNModelTrainer.evaluate_model(
                model=model,
                test_loader=test_loader,
                test_sample_loader=test_sample_loader,
                **model_config
            )
            print(f"Evaluate ACC: {acc}")

        case "DyGFormer":
            # TR sample 관련 파라미터
            seed=1
            batch_size=200
            max_hop=5
            n_pair=10
            n_sample=1000
            n_batch_sample=100

            data=DataUtils.preprocess_graph(dataset_name=f"enron")
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
            graph.set_random_seed(seed=seed)
            
            ### set data_loader, sample_loader
            train_df,val_df,test_df=TrainUtils.split_graph_df(df=graph_df)
            val_query_time=val_df["t"].max()
            test_query_time=test_df["t"].max()

            train_dataset=TemporalGraphDataset(df=train_df)
            val_dataset=TemporalGraphDataset(df=val_df)
            test_dataset=TemporalGraphDataset(df=test_df)

            train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
            val_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
            test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

            val_sample=graph.random_TR_sampling(
                n_sample=n_sample,
                n_pair=n_pair,
                query_time=val_query_time,
                max_hop=max_hop
            )
            print(f"Finish to generate val sample!")

            test_sample=graph.random_TR_sampling(
                n_sample=n_sample,
                n_pair=n_pair,
                query_time=test_query_time,
                max_hop=max_hop
            )
            print(f"Finish to generate test sample!")

            # Dataset
            val_sample_dataset=TRDataset(sample=val_sample,query_time=val_query_time)
            test_sample_dataset=TRDataset(sample=test_sample,query_time=test_query_time)

            # DataLoader, query_time 기준으로 생성되었기 때문에 shuffle 가능
            val_sample_loader=DataLoader(dataset=val_sample_dataset,batch_size=batch_size,shuffle=True)
            test_sample_loader=DataLoader(dataset=test_sample_dataset,batch_size=batch_size,shuffle=True)

            ### 하이퍼 파라미터
            n_layer=2
            n_neighbor=10
            n_head=4
            max_seq_len=10
            patch_size=5
            

            # 모델 학습 관련
            latent_dim=32
            time_dim=32
            output_dim=32
            co_dim=32
            common_dim=32
            epoch=100
            lr=0.0005
            optimizer=f"adam"
            early_stop=True
            patience=5

            ### model config
            model_config={
                "seed":seed,
                "batch_size":batch_size,
                "max_hop":max_hop,
                "n_pair":n_pair,
                "n_sample":n_sample,
                "n_batch_sample":n_batch_sample,
                "model_name":kwargs["model_name"],
                "n_layer":n_layer,
                "n_neighbor":n_neighbor,
                "n_head":n_head,
                "max_seq_len":max_seq_len,
                "patch_size":patch_size,
                "latent_dim":latent_dim,
                "time_dim":time_dim,
                "output_dim":output_dim,
                "co_dim":co_dim,
                "common_dim":common_dim,
                "epoch":epoch,
                "lr":lr,
                "optimizer":optimizer,
                "early_stop":early_stop,
                "patience":patience
            }

            ### set model and train
            model=DyGFormer(
                node_dim=node_dim,
                edge_dim=edge_dim,
                latent_dim=latent_dim,
                time_dim=time_dim,
                output_dim=output_dim,
                co_dim=co_dim,
                common_dim=common_dim,
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
                val_sample_loader=val_sample_loader,
                **model_config
            )
            acc=GNNModelTrainer.evaluate_model(
                model=model,
                test_loader=test_loader,
                test_sample_loader=test_sample_loader,
                **model_config
            )
            print(f"Evaluate ACC: {acc}")

if __name__=="__main__":
    """
    Execute test_fn
    """
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["TGAT","TGN","DyGFormer"],
        default=f"TGAT"
    )
    args=parser.parse_args()
    test_config={
        "model_name":args.model_name
    }
    test_fn(**test_config)