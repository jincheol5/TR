import argparse
import torch
from torch.utils.data import DataLoader
from utils import DataUtils,TrainUtils,TemporalGraphDataset
from graph import TemporalGraph,CTDNE_Graph
from model import CTDNE_TR
from model_train import ModelTrainer

"""
<< Test >> 
model_train.ModelTrainer
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test.
            model_train.ModelTrainer.train_TR
            Model: CTDNE
            """
            model_config={
                "model_name":"CTDNE",
                "walk_len":5,
                "min_walk_len":3,
                "n_walk":10,
                "n_window":100,
                "walk_epoch":10,
                "edge_sampling":"uniform",
                "neighbor_sampling":"uniform",
                "n_sample":400,
                "n_pair":10,
                "max_hop":5,
                "epoch":10,
                "lr":0.0005,
                "optimizer":"adam"
            }
            data=DataUtils.preprocess_graph(
                dataset_name=f"enron"
            )
            graph_df=data["graph_df"]
            graph=CTDNE_Graph(graph_df=graph_df)
            graph.set_random_seed(seed=1)
            model=CTDNE_TR(
                embed_dim=32,
                latent_dim=32,
                window_size=3,
                graph=graph
            )
            train_df,val_df,test_df=TrainUtils.split_graph_df(df=graph_df)
            train_dataset=TemporalGraphDataset(df=train_df)
            val_dataset=TemporalGraphDataset(df=val_df)
            test_dataset=TemporalGraphDataset(df=test_df)
            train_loader=DataLoader(dataset=train_dataset,batch_size=200,shuffle=False)
            val_loader=DataLoader(dataset=val_dataset,batch_size=200,shuffle=False)
            test_loader=DataLoader(dataset=test_dataset,batch_size=200,shuffle=False)

            """
            sample 생성 시 너무 오랜 시간 소모 -> 확인 필요
            """

            val_sample_loader=TrainUtils.get_TR_sample_loader(
                n_sample=400,
                n_pair=10,
                max_hop=5,
                data_loader=val_loader,
                graph=model.graph
            )

            test_sample_loader=TrainUtils.get_TR_sample_loader(
                n_sample=400,
                max_hop=5,
                data_loader=test_loader,
                graph=model.graph
            )

            model=ModelTrainer.train_walk_model(
                model=model,
                train_loader=train_loader,
                val_sample_loader=val_sample_loader,
                **model_config
            )
            ModelTrainer.evaluate_walk_model(
                model=model,
                sample_loader=test_sample_loader
            )


if __name__=="__main__":
    """
    Execute test_fn
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_num",type=int,default=1)
    args=parser.parse_args()
    test_config={
        'test_num':args.test_num
    }
    test_fn(**test_config)