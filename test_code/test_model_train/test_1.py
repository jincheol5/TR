import argparse
import torch
from torch.utils.data import DataLoader
from utils import DataUtils,TrainUtils,TemporalGraphDataset
from graph import TemporalGraph

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
                "walk_len":10,
                "min_walk_len":3,
                "n_context_window":100,
                "max_attempt":100,
                "walk_epoch":10,
                "edge_sampling":"uniform",
                "neighbor_sampling":"uniform",
                "epoch":10,
                "lr":0.0005,
                "seed":1,
                "optimizer":"adam"
            }

            data=DataUtils.preprocess_graph(
                dataset_name=f"enron"
            )
            graph_df=data["graph_df"]
            train_df,val_df,test_df=TrainUtils.split_graph_df(df=graph_df)
            train_dataset=TemporalGraphDataset(df=train_df)
            val_dataset=TemporalGraphDataset(df=val_df)
            test_dataset=TemporalGraphDataset(df=test_df)
            train_loader=DataLoader(dataset=train_dataset,batch_size=200,shuffle=False)
            val_loader=DataLoader(dataset=val_dataset,batch_size=200,shuffle=False)
            test_loader=DataLoader(dataset=test_dataset,batch_size=200,shuffle=False)


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