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
train_utils.TrainUtils
"""
def test_fn(**kwargs):
    match kwargs['test_num']:
        case 1:
            """
            Test. TrainUtils.get_sample_loader
            """
            data=DataUtils.preprocess_graph(dataset_name="enron")
            graph_df=data["graph_df"]
            graph=CTDNE_Graph(graph_df=graph_df)
            graph.set_random_seed(seed=1)

            train_df,val_df,test_df=TrainUtils.split_graph_df(df=graph_df)
            train_dataset=TemporalGraphDataset(df=train_df)
            val_dataset=TemporalGraphDataset(df=val_df)
            test_dataset=TemporalGraphDataset(df=test_df)

            train_loader=DataLoader(dataset=train_dataset,batch_size=200,shuffle=False)
            val_loader=DataLoader(dataset=val_dataset,batch_size=200,shuffle=False)
            test_loader=DataLoader(dataset=test_dataset,batch_size=200,shuffle=False)

            sample_laoder=TrainUtils.get_sample_loader(
                n_sample=100,
                n_pair=10,
                max_hop=5,
                data_loader=test_loader,
                graph=graph
            )

            print(len(sample_laoder))

        case 2:
            """
            Test. ModelTrainer.train_TR
            
            Model: CTDNE
            """
            model_name=f"CTDNE"
            seed=1

            data=DataUtils.preprocess_graph(dataset_name=f"enron")
            graph_df=data["graph_df"]
            graph=CTDNE_Graph(graph_df=graph_df)
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

            # TR sample 관련 파라미터
            n_sample=100
            n_pair=10
            max_hop=5 # walk_len과 동일해야 한다

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
                "model_name":model_name,
                "walk_len":walk_len,
                "min_walk_len":min_walk_len,
                "n_walk":n_walk,
                "n_window":n_window,
                "window_size":window_size,
                "edge_sampling":edge_sampling,
                "neighbor_sampling":neighbor_sampling,
                "walk_epoch":walk_epoch,
                "n_sample":n_sample,
                "n_pair":n_pair,
                "max_hop":max_hop,
                "embed_dim":embed_dim,
                "latent_dim":latent_dim,
                "epoch":epoch,
                "lr":lr,
                "optimizer":optimizer,
                "early_stop":early_stop,
                "patience":patience
            }

            ### set data_loader, sample_loader
            train_df,val_df,test_df=TrainUtils.split_graph_df(df=graph_df)
            train_dataset=TemporalGraphDataset(df=train_df)
            val_dataset=TemporalGraphDataset(df=val_df)
            test_dataset=TemporalGraphDataset(df=test_df)

            train_loader=DataLoader(dataset=train_dataset,batch_size=200,shuffle=False)
            val_loader=DataLoader(dataset=val_dataset,batch_size=200,shuffle=False)
            test_loader=DataLoader(dataset=test_dataset,batch_size=200,shuffle=False)

            val_sample_loader=TrainUtils.get_sample_loader(
                n_sample=n_sample,
                n_pair=n_pair,
                max_hop=max_hop,
                data_loader=val_loader,
                graph=graph
            )
            print(f"Finish to get val_sample_loader!")
            TrainUtils.check_sample_loader(sample_loader=val_sample_loader)

            test_sample_loader=TrainUtils.get_sample_loader(
                n_sample=n_sample,
                n_pair=n_pair,
                max_hop=max_hop,
                data_loader=test_loader,
                graph=graph
            )
            print(f"Finish to get test_sample_loader!")
            TrainUtils.check_sample_loader(sample_loader=test_sample_loader)

            ### set model and train
            model=CTDNE_TR(
                embed_dim=embed_dim,
                latent_dim=latent_dim,
                window_size=window_size,
                graph=graph
            )
            model=ModelTrainer.train_walk_model(
                model=model,
                train_loader=train_loader,
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
    parser.add_argument("--test_num",type=int,default=1)
    args=parser.parse_args()
    test_config={
        'test_num':args.test_num
    }
    test_fn(**test_config)