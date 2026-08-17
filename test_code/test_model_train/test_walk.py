import argparse
import torch
from torch.utils.data import DataLoader
from utils import DataUtils,TrainUtils,TRDataset
from graph import CTDNE_Graph,ATDGEB_Graph
from model import CTDNE_TR,ATDGEB_TR
from model_train import ModelTrainer

"""
<< Test >> 
model_train.ModelTrainer

Test Model:
    - CTDNE
    - ATDGEB
"""
def test_fn(**kwargs):
    match kwargs["model_name"]:
        case "CTDNE":
            """
            Test Model: CTDNE
            """
            seed=1
            data=DataUtils.preprocess_graph(dataset_name=f"enron")
            graph_df=data["graph_df"]
            train_df,val_df,test_df=TrainUtils.split_graph_df(df=graph_df)
            graph=CTDNE_Graph(graph_df=graph_df,train_df=train_df)
            graph.set_random_seed(seed=seed)
            n_node=graph.get_num_node()

            ### 하이퍼 파라미터
            # random walk 관련 파라미터
            walk_len=5
            min_walk_len=2
            n_walk=10
            n_window=int(n_walk*n_node*(walk_len-min_walk_len+1))
            window_size=2 # min_walk_len과 같아야 한다 
            edge_sampling=f"linear"
            neighbor_sampling=f"linear"
            walk_epoch=5

            # 모델 학습 관련
            embed_dim=32
            latent_dim=32
            epoch=100
            lr=0.0005
            optimizer=f"adam"
            early_stop=True
            patience=5

            ### model config
            model_config={
                "model_name":kwargs["model_name"],
                "walk_len":walk_len,
                "min_walk_len":min_walk_len,
                "n_walk":n_walk,
                "n_window":n_window,
                "window_size":window_size,
                "edge_sampling":edge_sampling,
                "neighbor_sampling":neighbor_sampling,
                "walk_epoch":walk_epoch,
                "embed_dim":embed_dim,
                "latent_dim":latent_dim,
                "epoch":epoch,
                "lr":lr,
                "optimizer":optimizer,
                "early_stop":early_stop,
                "patience":patience
            }

            ### set data_loader, sample_loader
            # sample
            train_query_time=train_df["t"].max()
            val_query_time=val_df["t"].max()
            test_query_time=test_df["t"].max()

            train_sample=graph.random_TR_sampling(
                n_sample=1000,
                n_pair=10,
                query_time=train_query_time,
                max_hop=5
            )
            print(f"Finish to generate train sample!")

            val_sample=graph.random_TR_sampling(
                n_sample=1000,
                n_pair=10,
                query_time=val_query_time,
                max_hop=5
            )
            print(f"Finish to generate val sample!")

            test_sample=graph.random_TR_sampling(
                n_sample=1000,
                n_pair=10,
                query_time=test_query_time,
                max_hop=5
            )
            print(f"Finish to generate test sample!")

            # Dataset
            train_sample_dataset=TRDataset(sample=train_sample)
            val_sample_dataset=TRDataset(sample=val_sample)
            test_sample_dataset=TRDataset(sample=test_sample)

            # DataLoader, query_time 기준으로 생성되었기 때문에 shuffle 가능
            train_sample_loader=DataLoader(dataset=train_sample_dataset,batch_size=100,shuffle=True)
            val_sample_loader=DataLoader(dataset=val_sample_dataset,batch_size=100,shuffle=True)
            test_sample_loader=DataLoader(dataset=test_sample_dataset,batch_size=100,shuffle=True)

            ### set model and train
            model=CTDNE_TR(
                embed_dim=embed_dim,
                latent_dim=latent_dim,
                window_size=window_size,
                graph=graph
            )
            model=ModelTrainer.train_walk_model(
                model=model,
                train_sample_loader=train_sample_loader,
                val_sample_loader=val_sample_loader,
                **model_config
            )
            acc=ModelTrainer.evaluate_walk_model(
                model=model,
                sample_loader=test_sample_loader
            )
            print(f"Evaluate ACC: {acc}")
        case "ATDGEB":
            """
            Test Model: ATDGEB
            """
            seed=1
            data=DataUtils.preprocess_graph(dataset_name=f"enron")
            graph_df=data["graph_df"]
            train_df,val_df,test_df=TrainUtils.split_graph_df(df=graph_df)
            graph=ATDGEB_Graph(graph_df=graph_df,train_df=train_df)
            graph.set_random_seed(seed=seed)
            n_node=graph.get_num_node()

            ### 하이퍼 파라미터
            # random walk 관련 파라미터
            k_list=[2,4,6,8,10]
            n_aggr=2
            min_points=3
            walk_len=5
            n_lsbs=3
            window_size=2 # min_walk_len과 같아야 한다 
            walk_epoch=5

            # 모델 학습 관련
            embed_dim=32
            latent_dim=32
            epoch=100
            lr=0.0005
            optimizer=f"adam"
            early_stop=True
            patience=5

            ### model config
            model_config={
                "model_name":kwargs["model_name"],
                "k_list":k_list,
                "n_aggr":n_aggr,
                "min_points":min_points,
                "walk_len":walk_len,
                "n_lsbs":n_lsbs,
                "window_size":window_size,
                "walk_epoch":walk_epoch,
                "embed_dim":embed_dim,
                "latent_dim":latent_dim,
                "epoch":epoch,
                "lr":lr,
                "optimizer":optimizer,
                "early_stop":early_stop,
                "patience":patience
            }

            ### set data_loader, sample_loader
            # sample
            train_query_time=train_df["t"].max()
            val_query_time=val_df["t"].max()
            test_query_time=test_df["t"].max()

            train_sample=graph.random_TR_sampling(
                n_sample=1000,
                n_pair=10,
                query_time=train_query_time,
                max_hop=5
            )
            print(f"Finish to generate train sample!")

            val_sample=graph.random_TR_sampling(
                n_sample=1000,
                n_pair=10,
                query_time=val_query_time,
                max_hop=5
            )
            print(f"Finish to generate val sample!")

            test_sample=graph.random_TR_sampling(
                n_sample=1000,
                n_pair=10,
                query_time=test_query_time,
                max_hop=5
            )
            print(f"Finish to generate test sample!")

            # Dataset
            train_sample_dataset=TRDataset(sample=train_sample)
            val_sample_dataset=TRDataset(sample=val_sample)
            test_sample_dataset=TRDataset(sample=test_sample)

            # DataLoader, query_time 기준으로 생성되었기 때문에 shuffle 가능
            train_sample_loader=DataLoader(dataset=train_sample_dataset,batch_size=100,shuffle=True)
            val_sample_loader=DataLoader(dataset=val_sample_dataset,batch_size=100,shuffle=True)
            test_sample_loader=DataLoader(dataset=test_sample_dataset,batch_size=100,shuffle=True)

            ### set model and train
            model=ATDGEB_TR(
                embed_dim=embed_dim,
                latent_dim=latent_dim,
                window_size=window_size,
                graph=graph
            )
            model=ModelTrainer.train_walk_model(
                model=model,
                train_sample_loader=train_sample_loader,
                val_sample_loader=val_sample_loader,
                **model_config
            )
            acc=ModelTrainer.evaluate_walk_model(
                model=model,
                sample_loader=test_sample_loader
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
        choices=["CTDNE","ATDGEB"],
        default=f"CTDNE"
    )
    args=parser.parse_args()
    test_config={
        "model_name":args.model_name
    }
    test_fn(**test_config)