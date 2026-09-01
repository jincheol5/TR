import argparse
from torch.utils.data import DataLoader
from graph import CTDNE_Graph,ATDGEB_Graph
from utils import DataUtils,TrainUtils,TemporalGraphDataset
from model import CTDNE,ATDGEB
from model_train import WalkModelTrainer

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
            data=DataUtils.preprocess_graph_dataset(dataset_name=f"enron")
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

            # TR 관련
            batch_size=200
            max_hop=5
            n_sample=1000
            n_pair=10

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
                "batch_size":batch_size,
                "max_hop":max_hop,
                "n_sample":n_sample,
                "n_pair":n_pair,
                "embed_dim":embed_dim,
                "latent_dim":latent_dim,
                "epoch":epoch,
                "lr":lr,
                "optimizer":optimizer,
                "early_stop":early_stop,
                "patience":patience
            }

            ### set query time
            val_query_time=val_df["t"].max()
            test_query_time=test_df["t"].max()

            ### set data_loader
            train_dataset=TemporalGraphDataset(df=train_df)
            train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)

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
            val_sample_list=TrainUtils.get_coarse_grained_TR_sample_list(
                n_node=n_node,
                n_sample=n_sample,
                n_pair=n_pair,
                query_time=val_query_time,
                batch_size=batch_size,
                TR_label=val_TR_label
            )
            test_TR_result=DataUtils.load_TR_result(
                dataset_name=f"enron",
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="test"
            )
            test_TR_label=test_TR_result["TR_label"]
            test_sample_list=TrainUtils.get_coarse_grained_TR_sample_list(
                n_node=n_node,
                n_sample=n_sample,
                n_pair=n_pair,
                query_time=test_query_time,
                batch_size=batch_size,
                TR_label=test_TR_label
            )

            ### set model and train
            model=CTDNE(
                embed_dim=embed_dim,
                latent_dim=latent_dim,
                window_size=window_size,
                graph=graph
            )
            model=WalkModelTrainer.train_model(
                model=model,
                train_loader=train_loader,
                val_sample_list=val_sample_list,
                TR_label=train_TR_label,
                **model_config
            )
            acc=WalkModelTrainer.evaluate_model(
                model=model,
                test_sample_list=test_sample_list,
                **model_config
            )
            print(f"Evaluate ACC: {acc}")

        case "ATDGEB":
            """
            Test Model: ATDGEB
            """
            seed=1
            data=DataUtils.preprocess_graph_dataset(dataset_name=f"enron")
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

            # TR 관련
            batch_size=200
            max_hop=5
            n_sample=1000
            n_pair=10

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
                "batch_size":batch_size,
                "max_hop":max_hop,
                "n_sample":n_sample,
                "n_pair":n_pair,
                "embed_dim":embed_dim,
                "latent_dim":latent_dim,
                "epoch":epoch,
                "lr":lr,
                "optimizer":optimizer,
                "early_stop":early_stop,
                "patience":patience
            }

            ### set query time
            val_query_time=val_df["t"].max()
            test_query_time=test_df["t"].max()

            ### set data_loader
            train_dataset=TemporalGraphDataset(df=train_df)
            train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)

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
            val_sample_list=TrainUtils.get_coarse_grained_TR_sample_list(
                n_node=n_node,
                n_sample=n_sample,
                n_pair=n_pair,
                query_time=val_query_time,
                batch_size=batch_size,
                TR_label=val_TR_label
            )
            test_TR_result=DataUtils.load_TR_result(
                dataset_name=f"enron",
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="test"
            )
            test_TR_label=test_TR_result["TR_label"]
            test_sample_list=TrainUtils.get_coarse_grained_TR_sample_list(
                n_node=n_node,
                n_sample=n_sample,
                n_pair=n_pair,
                query_time=test_query_time,
                batch_size=batch_size,
                TR_label=test_TR_label
            )

            ### set model and train
            model=ATDGEB(
                embed_dim=embed_dim,
                latent_dim=latent_dim,
                window_size=window_size,
                graph=graph
            )
            model=WalkModelTrainer.train_model(
                model=model,
                train_loader=train_loader,
                val_sample_list=val_sample_list,
                TR_label=train_TR_label,
                **model_config
            )
            acc=WalkModelTrainer.evaluate_model(
                model=model,
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
        choices=["CTDNE","ATDGEB"],
        default=f"CTDNE"
    )
    args=parser.parse_args()
    test_config={
        "model_name":args.model_name
    }
    test_fn(**test_config)